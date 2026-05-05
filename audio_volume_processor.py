#!/usr/bin/env python3
import os
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple
import math

class AudioVolumeProcessor:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.results = []

    def get_audio_stats(self, audio_path: Path) -> Dict[str, float | str]:
        cmd = [
            'ffmpeg', '-i', str(audio_path),
            '-af', 'volumedetect',
            '-f', 'null', '-'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = result.stderr + result.stdout

            mean_volume_match = re.search(r'mean_volume:\s*([-\d.]+)\s*dB', output)
            max_volume_match = re.search(r'max_volume:\s*([-\d.]+)\s*dB', output)

            mean_vol = float(mean_volume_match.group(1)) if mean_volume_match else -50.0
            max_vol = float(max_volume_match.group(1)) if max_volume_match else -50.0

            return {
                'mean': mean_vol,
                'max': max_vol,
                'file': audio_path.name,
                'path': str(audio_path)
            }
        except Exception as e:
            return {
                'mean': -50.0,
                'max': -50.0,
                'file': audio_path.name,
                'path': str(audio_path),
                'error': str(e)
            }

    def collect_wav_files(self) -> List[Path]:
        wav_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith('.wav') and '_enhanced' not in file:
                    wav_files.append(Path(root) / file)
        return sorted(wav_files)

    def analyze_volumes(self) -> List[Dict[str, float]]:
        print(f"正在扫描目录: {self.root_dir}")
        wav_files = self.collect_wav_files()
        print(f"找到 {len(wav_files)} 个 WAV 文件\n")

        self.results = []
        for i, wav_path in enumerate(wav_files, 1):
            print(f"[{i}/{len(wav_files)}] 正在分析: {wav_path.name}")
            stats = self.get_audio_stats(wav_path)
            self.results.append(stats)

        return self.results

    def print_statistics(self):
        if not self.results:
            print("没有可用的统计数据")
            return

        valid_results = [r for r in self.results if 'error' not in r]
        means = [r['mean'] for r in valid_results]
        maxes = [r['max'] for r in valid_results]

        print("\n" + "="*70)
        print("音量统计报告")
        print("="*70)
        print(f"{'文件名':<45} {'平均音量(dB)':<15} {'最大音量(dB)':<15}")
        print("-"*70)

        for r in self.results:
            error_msg = f" [错误: {r.get('error', 'unknown')}]" if 'error' in r else ""
            print(f"{r['file']:<45} {r['mean']:<15.2f} {r['max']:<15.2f}{error_msg}")

        print("-"*70)
        if means:
            print(f"\n统计摘要:")
            print(f"  平均音量范围: {min(means):.2f} dB ~ {max(means):.2f} dB")
            print(f"  平均值(Mean): {sum(means)/len(means):.2f} dB")
            print(f"  最大音量范围: {min(maxes):.2f} dB ~ {max(maxes):.2f} dB")
            print(f"  最大音量平均值: {sum(maxes)/len(maxes):.2f} dB")

            volume_range = max(means) - min(means)
            print(f"\n音量差异: {volume_range:.2f} dB")
            if volume_range > 6:
                print("⚠️  警告: 不同音频文件音量差异较大，建议进行标准化处理")
            else:
                print("✓  音量差异在可接受范围内")

    def get_loudnorm_params(self, input_path: Path) -> Dict[str, str] | None:
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json',
            '-f', 'null', '-'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            output = result.stderr + result.stdout

            i_match = re.search(r'"input_i"\s*:\s*"([-\d.]+)"', output)
            t_match = re.search(r'"input_tp"\s*:\s*"([-\d.]+)"', output)
            lra_match = re.search(r'"input_lra"\s*:\s*"([-\d.]+)"', output)
            thresh_match = re.search(r'"input_thresh"\s*:\s*"([-\d.]+)"', output)

            if i_match and thresh_match:
                return {
                    'input_i': i_match.group(1),
                    'input_tp': t_match.group(1) if t_match else '-1.5',
                    'input_lra': lra_match.group(1) if lra_match else '11',
                    'input_thresh': thresh_match.group(1)
                }
        except Exception:
            pass
        return None

    def enhance_audio(self, input_path: Path, output_path: Path, loudnorm_params: Dict[str, str]):
        cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-af', f'loudnorm=I=-16:TP=-1.5:LRA=11:'
                   f'measured_I={loudnorm_params["input_i"]}:'
                   f'measured_TP={loudnorm_params["input_tp"]}:'
                   f'measured_LRA={loudnorm_params["input_lra"]}:'
                   f'measured_thresh={loudnorm_params["input_thresh"]}:'
                   f'linear=true:print_format=summary',
            '-ar', '44100',
            '-ac', '2',
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0

    def normalize_and_enhance(self, output_suffix: str = '_enhanced'):
        if not self.results:
            print("请先运行 analyze_volumes() 进行统计分析")
            return

        valid_results = [r for r in self.results if 'error' not in r and '_enhanced' not in r['file']]
        if not valid_results:
            print("没有有效的音频文件可供处理")
            return

        print(f"\n{'='*70}")
        print(f"开始增强处理 (使用 loudnorm 响度标准化)")
        print(f"{'='*70}")
        print(f"目标响度: -16 LUFS (国际标准)")
        print(f"峰值限制: -1.5 dB")
        print(f"动态余量: 11 LU\n")

        success_count = 0
        fail_count = 0

        for i, result in enumerate(valid_results, 1):
            input_path = Path(result['path'])
            output_path = input_path.parent / f"{input_path.stem}{output_suffix}{input_path.suffix}"

            print(f"[{i}/{len(valid_results)}] 正在处理: {input_path.name}")

            params = self.get_loudnorm_params(input_path)
            if params:
                print(f"  测量响度: {params['input_i']} LUFS")
                if self.enhance_audio(input_path, output_path, params):
                    print(f"  ✓ 已保存: {output_path.name}")
                    success_count += 1
                else:
                    print(f"  ✗ 处理失败: {input_path.name}")
                    fail_count += 1
            else:
                print(f"  ✗ 无法获取音频参数: {input_path.name}")
                fail_count += 1

        print(f"\n处理完成: 成功 {success_count} 个, 失败 {fail_count} 个")
        print(f"增强后的文件已保存到相同目录，文件名添加了 '{output_suffix}' 后缀")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='批量处理音频文件音量统计和增强')
    parser.add_argument('directory', nargs='?', default='.',
                        help='要处理的目录路径 (默认: 当前目录)')
    parser.add_argument('--analyze-only', '-a', action='store_true',
                        help='仅进行分析统计，不进行增强处理')
    parser.add_argument('--target-db', '-t', type=float, default=-3.0,
                        help='目标音量dB值 (默认: -3.0)')
    parser.add_argument('--suffix', '-s', type=str, default='_enhanced',
                        help='输出文件后缀 (默认: _enhanced)')

    args = parser.parse_args()

    processor = AudioVolumeProcessor(args.directory)

    processor.analyze_volumes()
    processor.print_statistics()

    if not args.analyze_only:
        print("\n")
        processor.normalize_and_enhance(
            output_suffix=args.suffix
        )

if __name__ == '__main__':
    main()
