funclip/
├─ launch.py           # 启动 Gradio 服务的入口（参数：语言/端口/是否公网等）
├─ videoclipper.py     # ⭐核心：ASR+时间戳→文本/分段 → moviepy 视频裁剪/加字幕
├─ llm/
│   ├─ openai_api.py   # 接 OpenAI 兼容接口
│   ├─ qwen_api.py     # 接通义千问（OpenAI 兼容）
│   └─ g4f_openai_api.py
├─ utils/
│   ├─ subtitle_utils.py   # 生成整片 SRT 与片段 SRT（generate_srt, generate_srt_clip）
│   ├─ trans_utils.py      # 文本预处理、时间戳解析、AI剪辑结果时间戳抽取（extract_timestamps）
│   ├─ argparse_tools.py   # CLI 参数封装（命令行两阶段：识别/剪辑）
│   └─ theme.json          # Gradio 主题
└─ (可能还有 examples/, test/ 等示例与测试目录)
