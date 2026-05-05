#!/usr/bin/env python3
"""
Convert custom SRT file (with [Role: xxx] tags) to standard SRT file.
Usage: python convert_to_standard_srt.py
"""

import re
import os
from pathlib import Path


def parse_srt_time(time_str: str) -> float:
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def convert_srt(input_path: str, output_path: str = None):
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_standard.srt"

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n\n")

    output_lines = []
    for block in content:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        index_line = lines[0].strip()
        time_line = lines[1].strip()
        text_lines = lines[2:]

        text = "\n".join(text_lines)
        text = re.sub(r'\[Role:\s*[^\]]+\]\s*', '', text)
        text = text.strip()

        if not text:
            continue

        output_lines.append(index_line)
        output_lines.append(time_line)
        output_lines.append(text)
        output_lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Converted: {input_path}")
    print(f"Output: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert custom SRT to standard SRT")
    parser.add_argument("-i", "--input", default="segments_english/english.srt", help="Input SRT file")
    parser.add_argument("-o", "--output", default=None, help="Output SRT file (optional)")
    args = parser.parse_args()

    convert_srt(args.input, args.output)