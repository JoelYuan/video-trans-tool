import os
import sys
import argparse
import logging
import json
import re
from pathlib import Path
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_srt(srt_path: str) -> list:
    segments = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n\n")

    for block in content:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            try:
                index_line = lines[0].strip()
                time_line = lines[1].strip()
                text = "\n".join(lines[2:])

                index = index_line

                time_match = re.search(r'(\d{2}:\d{2}:\d{2}[,\.]\d+)\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d+)', time_line)
                if not time_match:
                    continue

                start_time = parse_srt_time(time_match.group(1))
                end_time = parse_srt_time(time_match.group(2))

                role = None
                clean_text = text
                role_match = re.search(r'\[角色:\s*([^\]]+)\]', time_line)
                if not role_match:
                    role_match = re.search(r'\[角色:\s*([^\]]+)\]', text)
                if role_match:
                    role = role_match.group(1).strip()
                    clean_text = re.sub(r'\[角色:\s*[^\]]+\]', '', text).strip()

                segments.append({
                    "index": index,
                    "start": start_time,
                    "end": end_time,
                    "text": clean_text,
                    "role": role,
                    "duration": end_time - start_time
                })
            except (ValueError, IndexError):
                continue

    return segments

def parse_srt_time(time_str: str) -> float:
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def extract_audio_segment(audio_path: str, start_time: float, end_time: float, output_path: str) -> bool:
    duration = end_time - start_time
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ss", str(start_time), "-t", str(duration),
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0

def group_by_role(segments: list) -> dict:
    roles = {}
    for seg in segments:
        role = seg.get("role")
        if role:
            if role not in roles:
                roles[role] = []
            roles[role].append(seg)
    return roles

def get_longest_segments(segments: list, count: int = 3) -> list:
    sorted_segs = sorted(segments, key=lambda x: x["duration"], reverse=True)
    return sorted_segs[:count]

def extract_segment_audio(audio_path: str, segments: list, output_dir: str) -> list:
    os.makedirs(output_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        start_time = seg["start"]
        end_time = seg["end"]
        duration = end_time - start_time

        output_path = os.path.join(output_dir, f"segment_{i+1:03d}.wav")
        if extract_audio_segment(audio_path, start_time, end_time, output_path):
            seg["audio_file"] = output_path
            logger.info(f"Extracted [{i+1}]: {output_path}")
        else:
            logger.warning(f"Failed to extract segment {i+1}")
            seg["audio_file"] = None

    return segments

def extract_reference_audio(audio_path: str, segments: list, output_dir: str, top_n: int = 3):
    roles = group_by_role(segments)
    logger.info(f"Found {len(roles)} roles: {', '.join(roles.keys())}")

    reference_info = {}

    for role, role_segments in roles.items():
        role_dir = os.path.join(output_dir, "reference", role)
        os.makedirs(role_dir, exist_ok=True)

        longest_segs = get_longest_segments(role_segments, top_n)

        logger.info(f"\nRole '{role}' - selecting {len(longest_segs)} longest segments:")
        for seg in longest_segs:
            logger.info(f"  - [{seg['start']:.2f}-{seg['end']:.2f}] ({seg['duration']:.2f}s): {seg['text'][:30]}...")

        role_files = []
        for i, seg in enumerate(longest_segs):
            ref_path = os.path.join(role_dir, f"ref_{i+1:02d}.wav")
            if extract_audio_segment(audio_path, seg["start"], seg["end"], ref_path):
                role_files.append({
                    "file": ref_path,
                    "text": seg["text"],
                    "duration": seg["duration"],
                    "start": seg["start"],
                    "end": seg["end"]
                })
                logger.info(f"  Saved: {ref_path}")

        reference_info[role] = role_files

    return reference_info

def main():
    parser = argparse.ArgumentParser(description="Extract audio segments from SRT file")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing audio.wav and original.srt")
    parser.add_argument("-s", "--srt", default="original.srt", help="SRT filename (default: original.srt)")
    parser.add_argument("-r", "--references", action="store_true", help="Also extract reference audio per role")
    parser.add_argument("-n", "--top-n", type=int, default=3, help="Number of longest segments per role (default: 3)")

    args = parser.parse_args()

    input_dir = Path(args.input)
    audio_path = input_dir / "audio.wav"
    srt_path = input_dir / args.srt

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        logger.info("Please run video-step1.py first to extract audio from video")
        return

    if not srt_path.exists():
        logger.error(f"SRT file not found: {srt_path}")
        return

    logger.info(f"Reading SRT file: {srt_path}")
    segments = parse_srt(str(srt_path))
    logger.info(f"Found {len(segments)} subtitle entries")

    segments_dir = input_dir / "segments"
    os.makedirs(segments_dir, exist_ok=True)

    if args.references:
        logger.info(f"\nExtracting reference audio (top {args.top_n} longest per role)...")
        reference_info = extract_reference_audio(str(audio_path), segments, str(segments_dir), args.top_n)

        ref_summary_path = segments_dir / "reference_summary.json"
        with open(ref_summary_path, "w", encoding="utf-8") as f:
            json.dump(reference_info, f, ensure_ascii=False, indent=2)
        logger.info(f"\nReference summary saved: {ref_summary_path}")

    logger.info(f"\nExtracting all audio segments to: {segments_dir}")
    segments = extract_segment_audio(str(audio_path), segments, str(segments_dir))

    metadata = {
        "source_dir": str(input_dir),
        "audio_path": str(audio_path),
        "srt_path": str(srt_path),
        "total_segments": len(segments),
        "segments": segments
    }

    metadata_path = segments_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    success_count = sum(1 for seg in segments if seg.get("audio_file"))
    logger.info(f"\nComplete!")
    logger.info(f"  - Extracted {success_count}/{len(segments)} segments")
    if args.references:
        logger.info(f"  - Reference audio saved in: {segments_dir}/reference/")
    logger.info(f"  - Output directory: {segments_dir}")

if __name__ == "__main__":
    main()
