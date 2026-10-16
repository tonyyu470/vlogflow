# VlogFlow

**把日常素材，剪成用你自己的声音讲述的故事。**

[English](README.en.md) · [使用说明](docs/usage.md) · [验证记录](docs/verification.md) · [问题反馈](https://github.com/tonyyu470/vlogflow/issues)

VlogFlow 是在自己电脑上启动、通过浏览器操作的开源视频创作应用。先审核脚本，再生成声音和成片。代码采用 MIT 许可，模型另行下载。

**v0.1.1 预览版**：已验证本地 Qwen3-VL 的五段素材分析与脚本生成、网页流程和合成素材渲染；新应用的真实 Qwen 配音至成片验收尚未完成。生成脚本仍需审核内容与口播长度，详见[验证记录](docs/verification.md)。

## 三步生成视频

1. **导入视频 → 审核脚本**：输入主题与事实，多文件上传或导入本机路径；分析带时间戳的代表画面，生成可编辑的逐镜头脚本。可以修改解说、来源视频、选段、顺序和环境声。
2. **单选字幕语言**：中文、英文、中英对照。字幕与配音语言独立；需要翻译时可生成并手动修改译文。
3. **选择你的音色 → 导出 MP4**：上传或填写参考音频路径，校正参考录音文字，试听、生成克隆配音、对齐字幕、剪辑和渲染。提供 MP4、SRT、最终脚本 JSON。

配音始终使用审核后的脚本文字。修改脚本会使旧结果失效；切换字幕可复用配音。配音长于允许选段时会明确报错，请缩短解说或扩大选段。应用不会自动删词、加速声音或生成对口型。

## 快速开始：Apple Silicon Mac

需要 Python **3.11 或 3.12**、FFmpeg、约 **12 GB** 模型与缓存磁盘空间。建议 **16 GB 或更多内存**；运行 MLX 前至少 **4.1 GiB 可用内存**。内存不足时任务会停止并保留中间结果，关闭不需要的应用后点击重试。

```bash
brew install ffmpeg ollama uv
git clone https://github.com/tonyyu470/vlogflow.git
cd vlogflow
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[mlx,asr]'
```

第一次运行时下载模型（数 GB，取决于网络）：

```bash
# 另开一个终端，启动本地视觉服务；不需要注册 API 账户
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 ollama serve
```

在已激活 VlogFlow 环境的终端继续：

```bash
ollama pull qwen3-vl:2b
vlogflow models download qwen-mlx
vlogflow models download asr
vlogflow doctor
vlogflow serve
```

浏览器打开 **http://127.0.0.1:7860**。安装后不需要 Node.js 或前端构建工具。也可使用 `python -m vlogflow.cli serve`。可选 `--port 7861 --no-browser`。

已有对应模型时，在「模型与设置 → 高级设置」填写本地目录，避免重复下载。模型必须匹配仓库附带的固定版本清单，程序会校验权重哈希。不要把另一种量化版本填进 MLX 模型目录。

## 模型与费用

| 模块 | 默认方案 | 执行位置 / 费用 |
|---|---|---|
| 画面分析、脚本、字幕翻译 | Ollama + Qwen3-VL 2B | 本地运行，不收 API 调用费 |
| 可选视觉服务 | OpenAI 兼容多模态 API | 抽样画面与需求发送到配置的服务，费用由提供方决定 |
| 音色克隆 | Qwen3-TTS 1.7B Base / MLX 8-bit | 本地运行，不收 API 调用费 |
| 参考转写、配音时间对齐 | faster-whisper base / CPU int8 | 本地运行 |
| 剪辑、字幕、音频合成 | FFmpeg + Pillow + Noto Sans SC | 本地运行 |

使用自配 API 时，在设置中填写 **包含 `/v1` 的 base URL**（如果提供方要求）、模型名称与 API Key。接口需要支持图片输入和 JSON 输出。不是所有宣称 OpenAI 兼容的接口都支持多模态；先测试连接，再分析一个短视频。

本地推理消耗电脑资源和电力；模型及第三方组件遵守各自许可证。请使用自己的声音或已获授权的参考录音。

## 运行与数据

- 服务只监听本机地址，不设计为多人或公网服务。不要直接通过端口转发公开此服务。
- macOS 数据默认位于 `~/Library/Application Support/VlogFlow`；Linux/Windows 使用系统应用数据目录。用 `VLOGFLOW_DATA_DIR` 可指定其他目录。
- 视频副本、录音、脚本版本、任务状态与结果保存在该目录。源码仓库不包含私人素材或模型权重。
- API Key 仅保存在本机配置，接口读回时隐藏。选择远程接口后，抽样画面、创作需求及翻译文本会发送给该提供方；参考录音和音色推理仍在本机。
- 重启后中断任务会标记为可重试。取消只终止本任务启动的工作进程；已完成缓存保留。
- 只有一条重型任务队列，视觉、ASR、TTS 和渲染不会由应用同时加载。请避免另一个程序同时运行大模型。

## 当前边界

这是 **v0.1.1 预览版**。实际验收和限制见 [验证记录](docs/verification.md)。

- 优先支持 Apple Silicon Mac。提供官方 Qwen PyTorch 适配器供 Windows/Linux 使用，**尚未在这些系统或 CUDA 上实机验证**，不是跨平台性能保证。安装方式见 [使用说明](docs/usage.md#其他系统实验性)。
- 画面理解使用每段三个代表帧，不能保证捕捉所有动作。金额、地点、时间等事实请写进需求并审核；它不等同于完整逐帧分析。
- 配置 ASR 后会检测每段前 30 秒的原声讲话并提示。请在脚本审核中预览素材，把有原口播的镜头设为「关闭原声」，避免双重讲话。第一版没有自动语音分离。
- 自动听写用于检查内容与对齐，不是音色相似度评分。参考录音质量、口音、跨语言和短句都会影响克隆效果。
- 支持中文和英文配音/字幕。混合语言、极短口播、很长单段解说可能需要手动调整；推荐每镜头 1–2 个简短句子。
- 每项目最多 50 个素材，每段不超过 10 分钟，单文件不超过 4 GiB；参考录音 3–30 秒、上传不超过 64 MiB。每批上传最多 8 GiB，大素材建议使用本机路径导入。
- 输出为 30 fps、短边 720 像素的 H.264/AAC MP4；混合画幅用留边适配，不做自动主体裁剪。

## 开发与测试

```bash
uv pip install -e '.[dev,asr]'
pytest
# 仅运行前端开发测试需要 Node.js 22+
node --test tests/frontend.test.cjs
python -m build
```

CI 使用程序生成的测试图形和音调，不下载大模型；真实模型验收单独记录。测试覆盖脚本边界、版本失效、旧任务保护、字幕语言复用、HTTP 接口、取消、资源保护和真实 FFmpeg 编解码。前端在 `src/vlogflow/static`，不需要构建步骤。

欢迎提交 Issue 和 PR。架构见 [architecture.md](docs/architecture.md)，第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
