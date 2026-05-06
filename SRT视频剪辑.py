#!/usr/bin/env python3
import os
import sys
import re
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_srt(srt_path: str) -> List[Tuple[float, float, str]]:
    segments = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if '-->' not in line:
            continue

        time_match = re.search(r'(\d{2}:\d{2}:\d{2},\d{2,3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{2,3})', line)
        if not time_match:
            continue

        start_str = time_match.group(1)
        end_str = time_match.group(2)

        start_ms = parse_time_to_ms(start_str)
        end_ms = parse_time_to_ms(end_str)
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0

        text_lines = []
        j = i + 1
        while j < len(lines) and lines[j].strip():
            line_content = lines[j].strip()
            if not line_content.startswith('[Role:'):
                text_lines.append(line_content)
            j += 1

        text = '\n'.join(text_lines)
        segments.append((start_sec, end_sec, text))

    return segments

def parse_time_to_ms(time_str: str) -> int:
    h, m, rest = time_str.split(':')
    s, ms = rest.split(',')
    ms = ms.zfill(3)
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def cut_video_segment(video_path: str, start_time: float, end_time: float, output_path: str, min_gap: float = 0.3) -> bool:
    duration = end_time - start_time

    if duration < min_gap:
        logger.debug(f"Segment too short ({duration:.2f}s < {min_gap}s), adjusting to minimum duration")
        end_time = start_time + min_gap

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(end_time - start_time),
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        return False

    return True

def concatenate_videos(segment_files: List[str], output_path: str) -> bool:
    if not segment_files:
        logger.warning("No video segments to concatenate")
        return False

    list_file = output_path + ".txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for seg_file in segment_files:
            f.write(f"file '{os.path.abspath(seg_file)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(list_file)

    if result.returncode != 0:
        logger.error(f"FFmpeg concat error: {result.stderr}")
        return False

    return True

def process_video_with_srt(video_path: str, srt_path: str, output_path: str = None, min_gap: float = 0.3) -> bool:
    video_path = Path(video_path)
    srt_path = Path(srt_path)

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return False

    if not srt_path.exists():
        logger.error(f"SRT file not found: {srt_path}")
        return False

    if output_path is None:
        output_path = str(video_path.stem) + "_srtcuts.mp4"

    segments = parse_srt(str(srt_path))
    if not segments:
        logger.error("No valid segments found in SRT file")
        return False

    logger.info(f"Found {len(segments)} segments in SRT file")

    output_dir = Path(output_path).parent / f"{video_path.stem}_srtcuts_temp"
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_files = []
    for i, (start, end, text) in enumerate(segments, 1):
        output_file = output_dir / f"seg_{i:03d}.mp4"
        logger.info(f"Cutting segment {i}/{len(segments)}: {format_time(start)} --> {format_time(end)}")

        if cut_video_segment(str(video_path), start, end, str(output_file), min_gap):
            segment_files.append(str(output_file))
        else:
            logger.warning(f"Failed to cut segment {i}")

    if not segment_files:
        logger.error("No segments were successfully cut")
        return False

    logger.info(f"\nConcatenating {len(segment_files)} video segments...")

    if concatenate_videos(segment_files, str(output_path)):
        logger.info(f"Successfully created: {output_path}")

        for seg_file in segment_files:
            os.remove(seg_file)
        output_dir.rmdir()

        return True
    else:
        logger.error("Failed to concatenate videos")
        return False

def main():
    parser = argparse.ArgumentParser(description="SRT视频剪辑 - 根据SRT字幕文件的时间点截取并拼接视频")
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("srt", help="SRT字幕文件路径")
    parser.add_argument("-o", "--output", default=None, help="输出视频路径 (默认: 视频名_srtcuts.mp4)")
    parser.add_argument("--min-gap", type=float, default=0.3, help="最小片段时长(秒)，短于此值的片段将延长至此长度 (默认: 0.3)")

    args = parser.parse_args()

    process_video_with_srt(args.video, args.srt, args.output, args.min_gap)

if __name__ == "__main__":
    main()
