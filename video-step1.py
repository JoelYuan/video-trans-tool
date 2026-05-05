import os
import sys
import re
import argparse
import subprocess
import logging
import json
from pathlib import Path
import numpy as np
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

def read_wav_file(wav_path: str) -> tuple:
    import wave
    with wave.open(wav_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
    
    if sample_width == 2:
        samples = np.frombuffer(data, dtype=np.int16)
    else:
        samples = np.frombuffer(data, dtype=np.int8)
    
    if n_channels > 1:
        samples = samples[::n_channels]
    
    return sample_rate, samples

def vad_segmentation(audio_path: str, aggressiveness: int = 3) -> list:
    import webrtcvad
    
    logger.info(f"Using WebRTC VAD (aggressiveness={aggressiveness})...")
    
    sample_rate, samples = read_wav_file(audio_path)
    
    vad = webrtcvad.Vad(aggressiveness)
    
    frame_duration_ms = 30
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    
    samples = np.pad(samples, (0, frame_size - (len(samples) % frame_size)), mode='constant')
    
    frames = []
    for i in range(0, len(samples), frame_size):
        frame = samples[i:i+frame_size].tobytes()
        is_speech = vad.is_speech(frame, sample_rate)
        frames.append({
            'is_speech': is_speech,
            'start': i / sample_rate,
            'end': (i + frame_size) / sample_rate
        })
    
    segments = []
    current_segment = None
    
    for frame in frames:
        if frame['is_speech']:
            if current_segment is None:
                current_segment = {
                    'start': frame['start'],
                    'end': frame['end']
                }
            else:
                current_segment['end'] = frame['end']
        else:
            if current_segment is not None:
                duration = current_segment['end'] - current_segment['start']
                if duration > 0.1:
                    segments.append(current_segment)
                current_segment = None
    
    if current_segment is not None:
        duration = current_segment['end'] - current_segment['start']
        if duration > 0.1:
            segments.append(current_segment)
    
    merged_segments = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            last = merged_segments[-1]
            gap = seg['start'] - last['end']
            if gap < 0.3:
                last['end'] = seg['end']
            else:
                merged_segments.append(seg)
    
    for seg in merged_segments:
        seg['text'] = ""
    
    return merged_segments

def asr_recognition(audio_path: str, segments: list, model_size: str = "base", device: str = "auto", 
                   model_path: str = None) -> list:
    from faster_whisper import WhisperModel
    
    logger.info(f"Loading Whisper model: {model_size if not model_path else model_path}...")
    
    if device == "auto":
        device = "cuda"
    
    compute_type = "float16"
    
    if model_path and os.path.exists(model_path):
        model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type
        )
    else:
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
    
    full_segments_list = list(full_segments)
    
    for seg in segments:
        start_time = seg["start"]
        end_time = seg["end"]
        
        text = ""
        for full_seg in full_segments_list:
            fs_start = full_seg.start
            fs_end = full_seg.end
            
            if fs_end > start_time and fs_start < end_time:
                overlap_start = max(start_time, fs_start)
                overlap_end = min(end_time, fs_end)
                overlap_ratio = (overlap_end - overlap_start) / (fs_end - fs_start)
                
                if overlap_ratio > 0.5:
                    text += full_seg.text + " "
        
        seg["text"] = text.strip()
        logger.debug(f"Segment [{start_time:.2f}-{end_time:.2f}]: {seg['text']}")
    
    return segments

def segments_to_srt(segments: list, output_srt: str):
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start_time = seg["start"]
            end_time = seg["end"]
            text = seg.get("text", "")

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
            f.write(f"[角色: 角色{i}]\n")
            f.write(f"{text}\n")
            f.write("\n")

    logger.info(f"SRT file saved: {output_srt}")

def segments_to_english_srt(segments: list, output_srt: str):
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start_time = seg["start"]
            end_time = seg["end"]
            text = seg.get("text", "")

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
            f.write(f"[Role: 角色{i}]\n")
            f.write(f"\n")
            f.write("\n")

    logger.info(f"English SRT file saved: {output_srt}")

def extract_segment_audio(audio_path: str, segments: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        start_time = seg["start"]
        end_time = seg["end"]
        duration = end_time - start_time

        output_path = os.path.join(output_dir, f"segment_{i+1:03d}.wav")
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start_time), "-t", str(duration),
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            seg["audio_file"] = output_path
            logger.info(f"Extracted: {output_path}")
        else:
            logger.warning(f"Failed to extract segment {i+1}: {result.stderr}")
            seg["audio_file"] = None

    return segments

def process_video(video_path: str, output_dir: str = None, vad_aggressiveness: int = 3, 
                 whisper_model: str = "base", whisper_device: str = "auto", 
                 whisper_model_path: str = None):
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return

    if output_dir is None:
        output_dir = str(video_path.stem + "_output")

    if os.path.exists(output_dir):
        import shutil
        logger.info(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    audio_wav = os.path.join(output_dir, "audio.wav")
    srt_path = os.path.join(output_dir, "original.srt")
    segments_dir = os.path.join(output_dir, "segments")

    logger.info(f"Processing video: {video_path}")

    if not extract_audio_from_video(str(video_path), audio_wav):
        return

    logger.info("Performing VAD segmentation...")
    segments = vad_segmentation(audio_wav, aggressiveness=vad_aggressiveness)
    logger.info(f"Found {len(segments)} speech segments")

    if not segments:
        logger.warning("No speech segments detected!")
        logger.info("You can try adjusting VAD aggressiveness with --vad-aggressiveness")
        return

    logger.info(f"Performing ASR recognition with Whisper...")
    segments = asr_recognition(audio_wav, segments, model_size=whisper_model, 
                               device=whisper_device, model_path=whisper_model_path)

    segments_to_srt(segments, srt_path)
    
    english_srt_path = os.path.join(output_dir, "english.srt")
    segments_to_english_srt(segments, english_srt_path)

    metadata = {
        "video_path": str(video_path),
        "audio_path": audio_wav,
        "total_segments": len(segments),
        "segments": segments
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"处理完成! 输出目录: {output_dir}")
    logger.info(f"  - audio.wav: 完整音频轨道")
    logger.info(f"  - original.srt: 字幕文件 (含 [角色: 角色N] 占位符)")
    logger.info(f"  - english.srt: 英文翻译字幕 (含 [Role: 角色N] 占位符)")
    logger.info(f"  - metadata.json: 处理元数据")
    logger.info(f"")
    logger.info(f"后续步骤:")
    logger.info(f"  1. 编辑 original.srt, 将 '角色N' 替换为实际角色名")
    logger.info(f"     (例如: [角色: wuwang], [角色: jiujianxian], [角色: 旁白])")
    logger.info(f"  2. 编辑 english.srt, 填入英文翻译文本")
    logger.info(f"  3. 运行: python video-step1.5.py -i {output_dir} -r -n 3")
    logger.info(f"  4. 运行: python video-step2.py -i {output_dir}")

    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Video Audio Extraction and VAD Segmentation with Whisper ASR")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("-o", "--output", default="segments_english", help="Output directory (default: segments_english)")
    parser.add_argument("--vad-aggressiveness", type=int, default=3, 
                        help="VAD aggressiveness (0-3, higher = more sensitive)")
    parser.add_argument("--whisper-model", type=str, default="medium",
                        help="Whisper model size: tiny, base, small, medium, large (default: medium)")
    parser.add_argument("--whisper-device", type=str, default="auto",
                        help="Device for Whisper: auto, cpu, cuda")
    parser.add_argument("--whisper-model-path", type=str, default=config.WHISPER_MODEL_PATH,
                        help="Path to local Whisper model directory")

    args = parser.parse_args()
    process_video(
        args.video, 
        args.output, 
        args.vad_aggressiveness,
        args.whisper_model,
        args.whisper_device,
        args.whisper_model_path
    )

if __name__ == "__main__":
    main()