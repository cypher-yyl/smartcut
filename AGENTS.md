# 仓库指南

## 项目结构与模块组织

核心自动化代码位于 `funclip/` 目录。`funclip/main.py` 启动 Gradio 界面，该界面通过 `videoclipper.py` 编排 ASR（自动语音识别）、语义分析和视频剪辑。LLM 桥接模块存放在 `funclip/llm/` 下，而 `funclip/tool/video_downloader.py` 负责远程媒体摄取，`funclip/utils/` 存放字幕/时间戳辅助工具。回归测试脚本位于 `funclip/test/`，文档和截图在 `docs/images/`，而 MoviePy 字幕使用的捆绑字体在 `font/`。

## 构建、测试和开发命令

在运行任何 GPU 密集型任务之前，请准备虚拟环境并安装已发布的依赖项：
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python funclip/main.py --lang zh --listen --share        # 启动 Gradio 工作区
python funclip/videoclipper.py --stage 1 --file <mp4>    # 第1步：ASR/转录
python funclip/videoclipper.py --stage 2 --file <mp4> --dest_text "片段" --output_dir ./output
bash funclip/test/test.sh                                # 脚本化的识别 + 剪辑
python funclip/test/imagemagick_test.py                  # 字幕渲染完整性检查
```

## 编码风格与命名规范

遵循 PEP 8 规范，使用 4 个空格缩进，并将行长度保持在 100 个字符左右，以确保回调函数保持可读性。使用小写下划线命名（`extract_timestamps`、`video_recog`），通过 `logging` 路由新的诊断信息，并优先使用字典或数据类来管理剪辑器状态。在 argparse 中使用小写标志公开新的 CLI 选项，并保持供应商特定的 LLM 逻辑封装，使 `_call_llm_any` 保持为唯一入口点。

## 测试指南

现有的回归覆盖是基于脚本的：`funclip/test/test.sh` 执行第 1 和第 2 阶段，`imagemagick_test.py` 验证字幕叠加。在提交 PR 之前运行这两个测试并捕获生成的产物。对于新的辅助工具，请在脚本旁边添加 `test_<feature>.py` 并将其接入 `python -m pytest funclip/test`，以便我们扩展自动化 CI。使用确定性的视频片段或缓存的 SRT 文件，并在代码注释中说明预期行为。

## 提交与拉取请求指南

提交历史使用简洁的命令式主题行（例如：`初始化`）。请以英文或中文保持这种风格，第一行不超过 50 个字符，如有需要可用简短正文扩展。每个 PR 应描述对用户的影响，列出您运行的命令，链接到跟踪的问题，并在 UI 流程发生变化时添加截图或 GIF（将其存储在 `docs/images/` 中）。

## 安全与配置提示

切勿硬编码凭据。`funclip/llm/openai_api.py` 读取 `OPENAI_API_KEY`，而 `funclip/llm/qwen_api.py` 需要您的 DashScope 密钥；请在 shell 中导出它们或保留一个被 git 忽略的 `.env` 文件。在下载前验证 URL，清理文件名，并将临时媒体存储在统一的输出目录下以便于清理。