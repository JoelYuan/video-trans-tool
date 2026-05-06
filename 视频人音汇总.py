import os
import sys
import re
import argparse
import subprocess
import logging
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def extract_audio_from_video(video_path: str, output_wav: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_wav
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        return False
    logger.info(f"Audio extracted to: {output_wav}")
    return True

def denoise_audio(input_wav: str, output_wav: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", input_wav,
        "-af", "highpass=f=200,lowpass=f=3000,arnndn=m=rnnoise-models/denoise.rnnn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_wav
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"FFmpeg denoise with rnnoise failed, trying basic filter")
        cmd = [
            "ffmpeg", "-y", "-i", input_wav,
            "-af", "highpass=f=200,lowpass=f=3000,highpass=f=80,lowpass=f=4000,afftdn=nf=-20:tn=-15,volume=2.0",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_wav
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"FFmpeg afftdn failed, trying simpler filter")
            cmd = [
                "ffmpeg", "-y", "-i", input_wav,
                "-af", "highpass=f=200,lowpass=f=3000,volume=2.0",
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                output_wav
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg basic denoise error: {result.stderr}")
                return False
    logger.info(f"Denoised audio saved to: {output_wav}")
    return True

def enhance_audio(input_path: str, output_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", "volume=2.0,highpass=f=200,lowpass=f=3000",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        logger.warning(f"FFmpeg enhance failed, trying simpler volume adjustment")
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", "volume=2.0",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"FFmpeg volume adjustment error: {result.stderr}")
            return False
    logger.debug(f"Enhanced audio saved to: {output_path}")
    return True

def asr_recognition(audio_path: str, model_size: str = "base", 
                   device: str = "auto", model_path: str = None) -> list:
    from faster_whisper import WhisperModel
    
    if device == "auto":
        device = "cuda"
    
    compute_type = "float16"
    
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading Whisper model from local path: {model_path}")
        model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type
        )
    elif os.path.exists(config.WHISPER_MODEL_PATH):
        logger.info(f"Loading Whisper model from config path: {config.WHISPER_MODEL_PATH}")
        model = WhisperModel(
            config.WHISPER_MODEL_PATH,
            device=device,
            compute_type=compute_type
        )
    else:
        logger.info(f"Loading Whisper model: {model_size}")
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
    
    logger.info("Performing ASR recognition on full audio...")
    
    full_segments, _ = model.transcribe(
        audio_path,
        language="zh",
        beam_size=5,
        word_timestamps=True
    )
    
    segments = []
    for full_seg in full_segments:
        segments.append({
            "start": full_seg.start,
            "end": full_seg.end,
            "text": full_seg.text.strip(),
            "speaker": None
        })
    logger.info(f"Created {len(segments)} segments from Whisper transcription")
    return segments

def segments_to_srt(segments: list, output_srt: str):
    speaker_to_role = {}
    role_counter = 1
    
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start_time = seg["start"]
            end_time = seg["end"]
            text = seg.get("text", "")
            speaker = seg.get("speaker")
            
            if speaker and speaker not in speaker_to_role:
                speaker_to_role[speaker] = role_counter
                role_counter += 1
            
            role_num = speaker_to_role.get(speaker, i)

            start_h = int(start_time // 3600)
            start_m = int((start_time % 3600) // 60)
            start_s = start_time % 60
            start_str = f"{start_h:02d}:{start_m:02d}:{start_s:05.2f}"

            end_h = int(end_time // 3600)
            end_m = int((end_time % 3600) // 60)
            end_s = end_time % 60
            end_str = f"{end_h:02d}:{end_m:02d}:{end_s:05.2f}"

            f.write(f"{i}\n")
            f.write(f"{start_str.replace('.', ',')} --> {end_str.replace('.', ',')}\n")
            f.write(f"[Role: 角色{role_num}]\n")
            f.write(f"{text}\n")
            f.write("\n")

    logger.info(f"SRT file saved: {output_srt}")
    if speaker_to_role:
        logger.info(f"Detected {len(speaker_to_role)} unique speakers")

def extract_audio_segment(audio_path: str, start_time: float, end_time: float, output_path: str) -> bool:
    duration = end_time - start_time
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ss", str(start_time), "-t", str(duration),
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        return False
    logger.debug(f"Extracted audio segment: {output_path}")
    return True

def concatenate_audio(segment_files: list, output_path: str) -> bool:
    if not segment_files:
        logger.warning("No audio segments to concatenate")
        return False
    
    list_file = output_path + ".txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for seg_file in segment_files:
            f.write(f"file '{os.path.abspath(seg_file)}'\n")
    
    cmd = [
        "ffmpeg", "-y", "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(list_file)
    
    if result.returncode != 0:
        logger.error(f"FFmpeg concat error: {result.stderr}")
        return False
    
    logger.info(f"Concatenated audio saved: {output_path}")
    return True

def process_video(video_path: str, output_dir: str = None, 
                 whisper_model: str = "medium", whisper_device: str = "auto", 
                 whisper_model_path: str = None, enable_denoise: bool = True,
                 enable_enhance: bool = True):
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return

    if output_dir is None:
        output_dir = str(video_path.stem + "_output")

    os.makedirs(output_dir, exist_ok=True)

    audio_wav = os.path.join(output_dir, "audio.wav")
    denoised_wav = os.path.join(output_dir, "audio_denoised.wav")
    srt_path = os.path.join(output_dir, "original.srt")
    wavcuts_original_dir = os.path.join(output_dir, "wavcuts_original")
    wavcuts_enhanced_dir = os.path.join(output_dir, "wavcuts_enhanced")
    concatenated_audio = os.path.join(output_dir, "concatenated.wav")
    concatenated_audio_enhanced = os.path.join(output_dir, "concatenated_enhanced.wav")

    logger.info(f"Processing video: {video_path}")

    if not extract_audio_from_video(str(video_path), audio_wav):
        return

    if enable_denoise:
        logger.info("Applying noise reduction...")
        if not denoise_audio(audio_wav, denoised_wav):
            denoised_wav = audio_wav
            logger.warning("Using original audio due to denoise failure")
        audio_for_asr = denoised_wav
    else:
        audio_for_asr = audio_wav

    logger.info("Performing ASR recognition with Whisper...")
    segments = asr_recognition(audio_for_asr, model_size=whisper_model, 
                               device=whisper_device, model_path=whisper_model_path)

    min_segment_duration = 0.5
    original_count = len(segments)
    segments = [seg for seg in segments if (seg["end"] - seg["start"]) >= min_segment_duration]
    if len(segments) < original_count:
        logger.info(f"Filtered {original_count - len(segments)} segments shorter than {min_segment_duration}s")
        logger.info(f"Remaining segments: {len(segments)}")

    segments_to_srt(segments, srt_path)

    os.makedirs(wavcuts_original_dir, exist_ok=True)
    os.makedirs(wavcuts_enhanced_dir, exist_ok=True)
    
    original_audio_files = []
    enhanced_audio_files = []
    original_success_count = 0
    enhanced_success_count = 0
    
    logger.info(f"\nExtracting audio segments...")
    
    for i, seg in enumerate(segments, 1):
        start_time = seg["start"]
        end_time = seg["end"]
        text = seg.get("text", "")[:20]
        text = text.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        
        original_output_filename = f"seg_{i:03d}_{text}.wav" if text else f"seg_{i:03d}.wav"
        original_output_path = os.path.join(wavcuts_original_dir, original_output_filename)
        
        enhanced_output_filename = f"seg_{i:03d}_{text}_enhanced.wav" if text else f"seg_{i:03d}_enhanced.wav"
        enhanced_output_path = os.path.join(wavcuts_enhanced_dir, enhanced_output_filename)
        
        temp_audio_path = os.path.join(wavcuts_enhanced_dir, f"temp_seg_{i:03d}.wav")
        
        if extract_audio_segment(audio_wav, start_time, end_time, original_output_path):
            original_audio_files.append(original_output_path)
            original_success_count += 1
            seg["original_audio_file"] = original_output_path
        else:
            seg["original_audio_file"] = None
        
        if extract_audio_segment(audio_wav, start_time, end_time, temp_audio_path):
            if enable_enhance:
                if enhance_audio(temp_audio_path, enhanced_output_path):
                    enhanced_audio_files.append(enhanced_output_path)
                    enhanced_success_count += 1
                else:
                    os.rename(temp_audio_path, enhanced_output_path)
                    enhanced_audio_files.append(enhanced_output_path)
                    enhanced_success_count += 1
            else:
                os.rename(temp_audio_path, enhanced_output_path)
                enhanced_audio_files.append(enhanced_output_path)
                enhanced_success_count += 1
            os.remove(temp_audio_path) if os.path.exists(temp_audio_path) else None
            seg["enhanced_audio_file"] = enhanced_output_path
        else:
            seg["enhanced_audio_file"] = None
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    logger.info(f"\nConcatenating original audio segments...")
    if concatenate_audio(original_audio_files, concatenated_audio):
        logger.info(f"Successfully created concatenated original audio")
    else:
        logger.warning("Failed to concatenate original audio segments")

    logger.info(f"\nConcatenating enhanced audio segments...")
    if concatenate_audio(enhanced_audio_files, concatenated_audio_enhanced):
        logger.info(f"Successfully created concatenated enhanced audio")
    else:
        logger.warning("Failed to concatenate enhanced audio segments")

    metadata = {
        "video_path": str(video_path),
        "audio_path": audio_wav,
        "denoised_audio_path": denoised_wav if enable_denoise else None,
        "srt_path": srt_path,
        "total_segments": len(segments),
        "extracted_original_audios": original_success_count,
        "extracted_enhanced_audios": enhanced_success_count,
        "segments": segments
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"\n处理完成! 输出目录: {output_dir}")
    logger.info(f"  - audio.wav: 提取的原始音频")
    if enable_denoise:
        logger.info(f"  - audio_denoised.wav: 降噪后的音频")
    logger.info(f"  - original.srt: 识别的字幕文件")
    logger.info(f"  - wavcuts_original/: 原始音频片段 ({original_success_count}/{len(segments)})")
    logger.info(f"  - wavcuts_enhanced/: 增强后的音频片段 ({enhanced_success_count}/{len(segments)})")
    logger.info(f"  - concatenated.wav: 拼接后的原始音频")
    logger.info(f"  - concatenated_enhanced.wav: 拼接后的增强音频")
    logger.info(f"  - metadata.json: 处理元数据")

    return output_dir

def main():
    parser = argparse.ArgumentParser(description="视频人音汇总 - 视频音频提取 + 降噪 + ASR识别 + 音频片段截取与增强")
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("-o", "--output", default=None, help="输出目录 (默认: 视频文件名_output)")
    parser.add_argument("--whisper-model", type=str, default=config.WHISPER_MODEL_SIZE,
                        help="Whisper模型大小: tiny, base, small, medium, large (默认: medium)")
    parser.add_argument("--whisper-device", type=str, default=config.WHISPER_DEVICE,
                        help="Whisper运行设备: auto, cpu, cuda")
    parser.add_argument("--whisper-model-path", type=str, default=config.WHISPER_MODEL_PATH,
                        help="本地Whisper模型路径 (默认: 从config.py读取)")
    parser.add_argument("--no-denoise", action="store_true", 
                        help="禁用降噪处理 (默认: 启用)")
    parser.add_argument("--no-enhance", action="store_true", 
                        help="禁用音频增强处理 (默认: 启用)")

    args = parser.parse_args()
    
    process_video(
        args.video, 
        args.output, 
        args.whisper_model,
        args.whisper_device,
        args.whisper_model_path,
        enable_denoise=not args.no_denoise,
        enable_enhance=not args.no_enhance
    )

if __name__ == "__main__":
    main()