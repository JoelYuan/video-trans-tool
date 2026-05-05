# 视频翻译器 - VoxCPM 工作流程

本项目使用 VoxCPM2 进行视频翻译配音，支持多角色音频提取和语音合成。

## 环境要求

```bash
pip install faster-whisper webrtcvad voxcpm scipy soundfile gradio_client numpy torch
```

### 系统依赖

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# 下载 https://ffmpeg.org/download.html 并添加到 PATH
```

## 模型下载

首次使用前需下载模型（使用 ModelScope 魔搭社区）：

```bash
# 安装 modelscope
pip install modelscope

# 下载 Whisper 模型
modelscope download --model pengzhendong/faster-whisper-medium --save ~/.cache/modelscope/hub/models/pengzhendong/faster-whisper-medium

# 下载 VoxCPM 模型
modelscope download --model OpenBMB/VoxCPM2 --save ~/.cache/modelscope/hub/models/OpenBMB/VoxCPM2
```

## 配置文件

`config.py` 中已配置模型路径，如有需要可自行修改：

```python
WHISPER_MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/pengzhendong/faster-whisper-medium")
VOXCPM_MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/OpenBMB/VoxCPM2")
```

## 工作流程总览

```
视频文件 (test.mp4)
    │
    ▼
Step 1: python video-step1.py test.mp4
    │  输出: segments_english/audio.wav, segments_english/original.srt
    ▼
Step 1.5: python video-step1.5.py -i segments_english -r -n 3
    │  输出: segments_english/segments/reference/{role}/ref_XX.wav
    ▼
Step 2: python video-step2.py -i segments_english
    │  输出: segments_english/voiceover.wav, segments_english/individual/
    ▼
完成!
```

---

## Step 1: video-step1.py

**功能**: 视频音频提取 + VAD分割 + Whisper ASR识别

```bash
python video-step1.py <视频文件>
```

### 参数说明
- 位置参数: 视频文件路径
- `-o, --output`: 输出目录 (默认: segments_english)
- `--vad-aggressiveness`: VAD敏感度 (0-3, 默认: 3)
- `--whisper-model`: Whisper模型大小 (默认: medium)
- `--whisper-model-path`: 本地Whisper模型路径

### 示例
```bash
python video-step1.py test.mp4
```

### 输出结构
```
segments_english/
├── audio.wav                              # 16kHz 单声道音频
├── original.srt                            # 带 [Role: xxx] 标签的 SRT (需手动编辑)
├── metadata.json                           # 处理元数据
└── segments/                               # 参考音频目录 (Step 1.5 生成)
    └── reference/
        ├── wuwang/ref_01.wav
        └── ...
```

### 注意事项
- `original.srt` 已自动添加 `[Role: 角色1]`、`[Role: 角色2]` 等占位标签，请手动修改为实际角色名

### SRT 角色标签格式
Step 1 自动生成带占位符的 SRT 文件，示例：
```
1
00:00:00,69 --> 00:00:03,21
[Role: 角色1]
这是第一句台词。
```

请将 `角色1` 修改为实际角色名（如 `wuwang`、`jiujianxian`、`旁白` 等）。

---

## Step 1.5: video-step1.5.py

**功能**: 从 SRT 提取各角色参考音频（最长片段）

```bash
python video-step1.5.py -i <输入目录> -r -n <每角色片段数>
```

### 参数说明
- `-i, --input`: 输入目录（包含 audio.wav 和 SRT 文件）
- `-s, --srt`: SRT 文件名 (默认: original.srt)
- `-r, --references`: 提取参考音频（必须加此参数）
- `-n, --top-n`: 每角色提取最长片段数 (默认: 3)

### 示例
```bash
python video-step1.5.py -i segments_english -r -n 3
```

### 输出结构
```
segments_english/
├── audio.wav
├── original.srt
└── segments/
    └── reference/                         # 角色参考音频
        ├── wuwang/
        │   ├── ref_01.wav
        │   ├── ref_02.wav
        │   └── ref_03.wav
        ├── jiujianxian/
        │   └── ref_01.wav
        └── ...
```

---

## Step 2: video-step2.py

**功能**: VoxCPM 多角色语音合成 + 混合输出

```bash
python video-step2.py -i <输入目录> -s <SRT文件>
```

### 参数说明
- `-i, --input`: 输入目录（包含 SRT 文件）
- `-s, --srt`: SRT 文件名 (默认: english.srt)
- `-r, --reference`: 参考音频目录 (默认: ../segments/segments/reference)
- `-o, --output`: 输出目录 (默认: segments_english)
- `--cfg`: CFG 引导强度 (默认: 2.0)
- `--steps`: 推理步数 (默认: 10)

### SRT 格式要求
SRT 文件需要包含角色标签：
```
1
00:00:00,69 --> 00:00:03,21
[Role: wuwang]
This is a capital offense.
```

### 示例
```bash
python video-step2.py -i segments_english
```

### 输出结构
```
segments_english/
├── voiceover.wav                          # 混合后的完整配音
├── metadata.json                          # 生成元数据
└── individual/                            # 离散音频文件
    ├── seg001_wuwang_This is a capital....wav
    ├── seg002_wuwang_This is a capital....wav
    └── ...
```

---

## 音频音量处理工具: audio_volume_processor.py

**功能**: 分析并标准化离散音频文件的音量

```bash
python audio_volume_processor.py <目录路径>
```

### 主要功能

1. **音量统计分析**: 扫描指定目录下的所有 WAV 文件，分析每个文件的平均音量和最大音量
2. **统计报告生成**: 输出详细的音量统计报告，包括音量范围、平均值等
3. **响度标准化参数计算**: 根据 EBU R128 标准计算 loudnorm 参数，用于音频标准化

### 使用场景

在 Step 2 生成 `individual/` 目录下的离散音频文件后，可使用此工具：
- 检查所有分段音频的音量是否一致
- 计算标准化参数以统一音频响度
- 识别音量异常的音频片段

### 示例

```bash
# 分析 segments_english/individual/ 目录下的音频文件
python audio_volume_processor.py segments_english/individual

# 或分析整个输出目录
python audio_volume_processor.py segments_english
```

### 输出示例

```
正在扫描目录: segments_english/individual
找到 15 个 WAV 文件

[1/15] 正在分析: seg001_wuwang_xxx.wav
...

==============================================================
音量统计报告
==============================================================
文件名                                         平均音量(dB)    最大音量(dB)
----------------------------------------------------------------------
seg001_wuwang_xxx.wav                         -23.50         -3.20
...

----------------------------------------------------------------------

统计摘要:
  平均音量范围: -25.30 dB ~ -20.10 dB
  平均值(Mean): -22.85 dB
  最大音量范围: -5.00 dB ~ -1.20 dB
  最大音量平均值: -2.85 dB

音量差异: 5.20 dB
⚠️  警告: 不同音频文件音量差异较大，建议进行标准化处理
```

### 配合 ffmpeg 标准化音频

根据输出的 `get_loudnorm_params` 参数，使用 ffmpeg 进行标准化：

```bash
ffmpeg -i input.wav -af loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json -f null -
```

---

## 完整使用示例

```bash
# 1. 提取音频 + ASR识别
python video-step1.py test.mp4

# 2. 编辑 original.srt，将 角色N 替换为实际角色名
# 例如: [Role: 角色1] -> [Role: wuwang]

# 3. 提取角色参考音频
python video-step1.5.py -i segments_english -r -n 3

# 4. 生成配音
python video-step2.py -i segments_english
```

---

## 测试脚本

### test_whisper.py
测试 Whisper ASR 模型
```bash
python test_whisper.py
```

### test_step2_single.py
测试 VoxCPM 单句生成
```bash
python test_step2_single.py
```

---

## 常见问题

### Q: torchcodec 加载错误
**A**: PyTorch 2.9 与 torchcodec 不兼容。video-step2.py 已设置 `load_denoiser=False` 禁用 ZipEnhancer 降噪功能。

### Q: 如何调整生成音频的质量？
**A**: 增大 `--steps` 参数（默认10，建议10-30），或增大 `--cfg` 参数（默认2.0，范围1.0-3.0）。

### Q: VAD 检测不到语音
**A**: 尝试降低 VAD 敏感度: `--vad-aggressiveness 2` 或 `1`

---

## 目录结构最终示例

```
video-trans-tool/
├── config.py                              # 配置文件（模型路径）
├── video-step1.py
├── video-step1.5.py
├── video-step2.py
├── test_whisper.py
├── test_step2_single.py
├── segments_english/
│   ├── audio.wav
│   ├── original.srt        # 带 [Role: 角色N] 占位标签
│   ├── english.srt         # 编辑后的英文SRT
│   ├── voiceover.wav
│   ├── metadata.json
│   ├── individual/
│   │   ├── seg001_角色1_xxx.wav
│   │   └── ...
│   └── segments/
│       └── reference/
│           ├── 角色1/ref_01.wav
│           ├── 角色2/ref_01.wav
│           └── ...
└── ...
```