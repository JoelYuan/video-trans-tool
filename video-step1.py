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
from collections import defaultdict

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

def speaker_diarization(audio_path: str, model_path: str = None) -> list:
    """使用 pyannote-segmentation-3-0 进行说话人分割"""
    import onnxruntime as ort
    import wave
    
    # 如果没有提供模型路径，使用默认路径
    if model_path is None:
        model_path = config.PYANNOTE_SEGMENTATION_MODEL_PATH
    
    # 检查模型文件
    model_file = os.path.join(model_path, "model.onnx")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"ONNX model file not found: {model_file}")
    
    logger.info(f"Using pyannote-segmentation model: {model_path}")
    
    # 使用 ONNX Runtime 加载模型
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(model_file, providers=providers)
    
    # 使用 wave 模块读取音频（与 test_whisper.py 保持一致）
    with wave.open(audio_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            samples = samples[::n_channels]
    
    logger.info(f"Audio loaded: {len(samples)} samples at {sample_rate} Hz")
    
    # pyannote-segmentation-3.0 期望输入格式: (1, num_channels, num_samples)
    input_audio = samples[np.newaxis, np.newaxis, :]
    
    logger.info("Running speaker segmentation inference...")
    outputs = session.run(None, {session.get_inputs()[0].name: input_audio})
    
    speaker_segments = []
    if outputs:
        logits = outputs[0]
        logger.info(f"Model output shape: {logits.shape}")
        
        # 检查输出维度，确定正确的解析方式
        if len(logits.shape) == 3:
            batch, dim1, dim2 = logits.shape
            
            # 判断哪个维度是帧数，哪个是说话人数
            # pyannote-segmentation-3.0 通常输出: (1, num_frames, num_speakers+1)
            # 但有时也可能是: (1, num_speakers+1, num_frames)
            if dim1 < dim2:
                # 可能是 (1, num_speakers+1, num_frames) 格式
                num_speakers = dim1 - 1
                num_frames = dim2
                frame_dim = 2
                speaker_dim = 1
                logger.info(f"Detected format: (batch, speakers+1, frames)")
            else:
                # 可能是 (1, num_frames, num_speakers+1) 格式
                num_frames = dim1
                num_speakers = dim2 - 1
                frame_dim = 1
                speaker_dim = 2
                logger.info(f"Detected format: (batch, frames, speakers+1)")
            
            logger.info(f"Model output: {num_frames} frames, {num_speakers} speakers")
            
            # pyannote-segmentation-3.0 输出步长是 10ms
            frame_duration = 0.01
            
            # 使用较低的阈值来检测说话人
            threshold = 0.2
            logger.info(f"Using detection threshold: {threshold}")
            
            total_frames_with_speech = 0
            for i in range(num_frames):
                frame_start = i * frame_duration
                frame_end = (i + 1) * frame_duration
                
                # 根据检测到的维度格式获取概率
                if frame_dim == 1:
                    speaker_probs = logits[0, i, :-1]
                else:
                    speaker_probs = logits[0, :-1, i]
                
                max_prob = np.max(speaker_probs)
                
                if max_prob > threshold:
                    total_frames_with_speech += 1
                    speaker_id = np.argmax(speaker_probs)
                    speaker_segments.append({
                        'start': frame_start,
                        'end': frame_end,
                        'speaker': f'speaker_{speaker_id}'
                    })
            
            logger.info(f"Detected speech in {total_frames_with_speech}/{num_frames} frames")
    
    # 合并连续的相同说话人片段
    if speaker_segments:
        merged = []
        current = speaker_segments[0].copy()
        for seg in speaker_segments[1:]:
            if seg['speaker'] == current['speaker']:
                current['end'] = seg['end']
            else:
                merged.append(current)
                current = seg.copy()
        merged.append(current)
        speaker_segments = merged
    
    # 过滤短片段
    speaker_segments = merge_short_segments(speaker_segments, min_duration=0.3)
    
    if speaker_segments:
        speaker_count = len(set(s['speaker'] for s in speaker_segments))
        logger.info(f"Found {len(speaker_segments)} speaker segments from {speaker_count} speakers")
    else:
        logger.info("No speaker segments detected")
    
    return speaker_segments

def merge_short_segments(segments: list, min_duration: float = 0.5) -> list:
    merged = []
    for seg in segments:
        duration = seg['end'] - seg['start']
        if duration >= min_duration:
            merged.append(seg)
        elif merged:
            merged[-1]['end'] = seg['end']
    return merged

def merge_vad_with_speaker_segments(vad_segments: list, speaker_segments: list) -> list:
    if not speaker_segments:
        return vad_segments
    
    result_segments = []
    
    for vad_seg in vad_segments:
        vad_start = vad_seg['start']
        vad_end = vad_seg['end']
        
        speaker_info = []
        for speaker_seg in speaker_segments:
            sp_start = speaker_seg['start']
            sp_end = speaker_seg['end']
            
            overlap_start = max(vad_start, sp_start)
            overlap_end = min(vad_end, sp_end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                speaker_info.append({
                    'speaker': speaker_seg['speaker'],
                    'duration': overlap_duration
                })
        
        if speaker_info:
            speaker_info.sort(key=lambda x: x['duration'], reverse=True)
            dominant_speaker = speaker_info[0]['speaker']
        else:
            dominant_speaker = None
        
        new_seg = vad_seg.copy()
        new_seg['speaker'] = dominant_speaker
        result_segments.append(new_seg)
    
    final_segments = []
    for seg in result_segments:
        if not final_segments:
            final_segments.append(seg)
        else:
            last = final_segments[-1]
            
            same_speaker = seg.get('speaker') == last.get('speaker')
            gap = seg['start'] - last['end']
            
            if same_speaker and gap < 0.3:
                last['end'] = seg['end']
                if 'text' in seg:
                    last['text'] = (last.get('text', '') + ' ' + seg.get('text', '')).strip()
            else:
                final_segments.append(seg)
    
    return final_segments

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
    
    # 如果没有输入片段，直接使用 Whisper 的转录结果作为片段
    if segments is None:
        segments = []
        for full_seg in full_segments_list:
            segments.append({
                "start": full_seg.start,
                "end": full_seg.end,
                "text": full_seg.text.strip(),
                "speaker": None
            })
        logger.info(f"Created {len(segments)} segments from Whisper transcription")
        return segments
    
    # 否则，将 Whisper 结果映射到现有片段
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
                 whisper_model_path: str = None, enable_diarization: bool = True):
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

    if enable_diarization:
        logger.info("Performing speaker diarization...")
        speaker_segments = speaker_diarization(audio_wav)
        
        if speaker_segments:
            logger.info("Merging VAD segments with speaker information...")
            segments = merge_vad_with_speaker_segments(segments, speaker_segments)
            logger.info(f"After merging: {len(segments)} segments")
        else:
            logger.info("Speaker diarization failed, falling back to Whisper-based segmentation")
            # 使用 Whisper 的转录结果作为片段（保留所有识别到的文本）
            segments = None

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
    logger.info(f"  - original.srt: 字幕文件 (含 [Role: 角色N] 占位符)")
    logger.info(f"  - english.srt: 英文翻译字幕 (含 [Role: 角色N] 占位符)")
    logger.info(f"  - metadata.json: 处理元数据")
    logger.info(f"")
    logger.info(f"后续步骤:")
    logger.info(f"  1. 编辑 original.srt, 将 '角色N' 替换为实际角色名")
    logger.info(f"     (例如: [Role: wuwang], [Role: jiujianxian], [Role: 旁白])")
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
    parser.add_argument("--no-diarization", action="store_true", 
                        help="Disable speaker diarization (default: enabled)")

    args = parser.parse_args()
    process_video(
        args.video, 
        args.output, 
        args.vad_aggressiveness,
        args.whisper_model,
        args.whisper_device,
        args.whisper_model_path,
        enable_diarization=not args.no_diarization
    )

if __name__ == "__main__":
    main()