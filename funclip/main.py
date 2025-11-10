"""
基础版本 + 语义理解与摘要|SemanticAnalysis + 通过URL下载视频（模块化）
此版本未结合：语义理解与摘要|SemanticAnalysisLLM 智能裁剪｜LLM Clipping
"""


import os
import re
import json
import logging
import argparse
import gradio as gr
from funasr import AutoModel                 # FunASR 统一入口：ASR/VAD/标点/说话人
from videoclipper import VideoClipper        # 识别、裁剪、字幕、导出
from llm.openai_api import openai_call       # OpenAI/Deepseek/Moonshot
from llm.qwen_api import call_qwen_model     # Qwen 系列
from llm.g4f_openai_api import g4f_openai_call  # 免费代理类接口
from utils.trans_utils import extract_timestamps
from introduction import top_md_1, top_md_3, top_md_4


from tool.video_downloader import (
    download_video_action,
    resolve_video_input,
)


# ============== 语义理解与摘要分析：Prompt ==============
PROMPT_SEMANTIC_FULL = """你是一个视频摘要与剪辑策划助手。输入为视频的完整 SRT 字幕（含时间与文本）。请基于内容完成：
1) 归纳视频主题；
2) 提取视频中的关键词/关键短语（含专有名词）；
3) 输出结构化大纲（若能从内容推断章节，请给出每章的大致起止时间）；
4) 识别视频中的“精彩片段”，要求：语义完整、时间连续、能代表核心观点或情绪高潮；每条给出起止时间(秒)、0-1 置信度分、简要理由、引用一句原文。
5) 把专有名词按类别分到 entities: {person, org, product, tech_term}。
6) 如有对后续剪辑有用的提示（节奏、转折、结论），写到 notes。

严格输出单个 JSON，不要解释或夹杂多余文本。时间单位使用“秒”，小数保留两位。

以下是 SRT 文本：
"""

def _safe_json_parse(s: str) -> dict:
    """从 LLM 输出中提取 JSON；失败则尝试首尾花括号截取。"""
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            head, tail = s.find("{"), s.rfind("}")
            if head != -1 and tail != -1 and head < tail:
                return json.loads(s[head:tail+1])
        except Exception:
            pass
        raise ValueError("LLM 未返回合法 JSON。")

def _call_llm_any(apikey: str, model: str, prompt: str, content: str) -> str:
    """统一的 LLM 调用封装：qwen / gpt / moonshot / deepseek / g4f-*"""
    if model.startswith("qwen"):
        return call_qwen_model(apikey, model, user_input=prompt + content, system_input="")
    if model.startswith("gpt") or model.startswith("moonshot") or model.startswith("deepseek"):
        return openai_call(apikey, model, system_content="", user_content=prompt + content)
    if model.startswith("g4f"):
        pure_model = "-".join(model.split("-")[1:])
        return g4f_openai_call(pure_model, "", prompt + content)
    raise ValueError("Unsupported model prefix. Use one of: qwen / gpt / g4f / moonshot / deepseek")

def semantic_analysis_run(srt_text: str, apikey: str, model: str) -> str:
    """
    语义理解与摘要分析主入口：
    - 输入：SRT 文本（带时间与内容）
    - 输出：JSON 字符串（包含 topics/keywords/outline/entities/highlights/notes）
    """
    if not srt_text or not srt_text.strip():
        return json.dumps({"error": "empty srt"}, ensure_ascii=False, indent=2)
    try:
        raw = _call_llm_any(apikey, model, PROMPT_SEMANTIC_FULL, srt_text)
        data = _safe_json_parse(raw)

        # 兜底字段
        data.setdefault("topics", [])
        data.setdefault("keywords", [])
        data.setdefault("outline", [])
        data.setdefault("entities", {"person": [], "org": [], "product": [], "tech_term": []})
        data.setdefault("highlights", [])
        data.setdefault("notes", "")

        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.exception(e)
        return json.dumps({
            "topics": [], "keywords": [], "outline": [],
            "entities": {"person": [], "org": [], "product": [], "tech_term": []},
            "highlights": [], "notes": f"fallback: {str(e)}"
        }, ensure_ascii=False, indent=2)

def highlights_to_timestamps(analysis_json_str: str):
    """
    从语义分析 JSON 中抽取高光片段为 timestamp_list [[start, end], ...]（单位：秒）
    - 过滤长度 < 2s 的过短片段
    - 最多返回 6 段
    """
    try:
        data = json.loads(analysis_json_str)
    except Exception:
        return []
    ts = []
    for h in data.get("highlights", []):
        try:
            s = float(h.get("start", 0))
            e = float(h.get("end", 0))
            if e > s and (e - s) >= 2.0:
                ts.append([round(s, 2), round(e, 2)])
        except Exception:
            continue
    return ts[:6]

# ============= 新增: 兼容旧版 video_clip 的时间单位缩放 =============
def _compat_scale_seconds_for_legacy(ts_list):
    """
    传入: [[start_sec, end_sec], ...]  (单位: 秒)
    目的: 兼容底层 video_clip/clip 中 'x16 再 /16000' 的旧逻辑。
         为了让最终日志仍然显示为秒，我们在这里将秒转成"毫秒"传下去：
         (sec * 1000) * 16 / 16000 = sec
    返回: [[start_scaled, end_scaled], ...]  (单位: 传给底层的数值)
    """
    scaled = []
    for s, e in ts_list:
        scaled.append([s * 1000.0, e * 1000.0])
    return scaled
# ======================================================

# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--lang', '-l', type=str, default="zh", help="language")
    parser.add_argument('--share', '-s', action='store_true', help="if to establish gradio share link")
    parser.add_argument('--port', '-p', type=int, default=7860, help='port number')
    parser.add_argument('--listen', action='store_true', help="if to listen to all hosts")
    args = parser.parse_args()

    # ===== ASR 模型装载：中文/英文两套 =====
    if args.lang == 'zh':
        funasr_model = AutoModel(
            model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
        )
    else:
        funasr_model = AutoModel(
            model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
        )

    audio_clipper = VideoClipper(funasr_model)
    audio_clipper.lang = args.lang

    server_name = '127.0.0.1'
    if args.listen:
        server_name = '0.0.0.0'

    # ======================
    # ===== 回调函数区 =====
    # ======================

    def audio_recog(audio_input, sd_switch, hotwords, output_dir):
        return audio_clipper.recog(audio_input, sd_switch, None, hotwords, output_dir=output_dir)

    def video_recog(video_input, sd_switch, hotwords, output_dir):
        return audio_clipper.video_recog(video_input, sd_switch, hotwords, output_dir=output_dir)

    def video_clip(dest_text, video_spk_input, start_ost, end_ost, state, output_dir):
        return audio_clipper.video_clip(
            dest_text, start_ost, end_ost, state, dest_spk=video_spk_input, output_dir=output_dir
        )

    def mix_recog(video_input, audio_input, hotwords, output_dir):
        output_dir = output_dir.strip()
        output_dir = os.path.abspath(output_dir) if output_dir else None
        audio_state, video_state = None, None
        if video_input is not None:
            res_text, res_srt, video_state = video_recog(video_input, 'No', hotwords, output_dir=output_dir)
            return res_text, res_srt, video_state, None
        if audio_input is not None:
            res_text, res_srt, audio_state = audio_recog(audio_input, 'No', hotwords, output_dir=output_dir)
            return res_text, res_srt, None, audio_state

    def mix_recog_speaker(video_input, audio_input, hotwords, output_dir):
        output_dir = output_dir.strip()
        output_dir = os.path.abspath(output_dir) if output_dir else None
        audio_state, video_state = None, None
        if video_input is not None:
            res_text, res_srt, video_state = video_recog(video_input, 'Yes', hotwords, output_dir=output_dir)
            return res_text, res_srt, video_state, None
        if audio_input is not None:
            res_text, res_srt, audio_state = audio_recog(audio_input, 'Yes', hotwords, output_dir=output_dir)
            return res_text, res_srt, None, audio_state

    def mix_clip(dest_text, video_spk_input, start_ost, end_ost, video_state, audio_state, output_dir):
        output_dir = output_dir.strip()
        output_dir = os.path.abspath(output_dir) if output_dir else None
        if video_state is not None:
            clip_video_file, message, clip_srt = audio_clipper.video_clip(
                dest_text, start_ost, end_ost, video_state, dest_spk=video_spk_input, output_dir=output_dir)
            return clip_video_file, None, message, clip_srt
        if audio_state is not None:
            (sr, res_audio), message, clip_srt = audio_clipper.clip(
                dest_text, start_ost, end_ost, audio_state, dest_spk=video_spk_input, output_dir=output_dir)
            return None, (sr, res_audio), message, clip_srt

    def video_clip_addsub(dest_text, video_spk_input, start_ost, end_ost, state, output_dir, font_size, font_color):
        output_dir = output_dir.strip()
        output_dir = os.path.abspath(output_dir) if output_dir else None
        return audio_clipper.video_clip(
            dest_text, start_ost, end_ost, state,
            font_size=font_size, font_color=font_color,
            add_sub=True, dest_spk=video_spk_input, output_dir=output_dir
        )

    def llm_inference(system_content, user_content, srt_text, model, apikey):
        SUPPORT_LLM_PREFIX = ['qwen', 'gpt', 'g4f', 'moonshot', 'deepseek']
        if model.startswith('qwen'):
            return call_qwen_model(apikey, model, user_content+'\n'+srt_text, system_content)
        if model.startswith('gpt') or model.startswith('moonshot') or model.startswith('deepseek'):
            return openai_call(apikey, model, system_content, user_content+'\n'+srt_text)
        elif model.startswith('g4f'):
            model2 = "-".join(model.split('-')[1:])
            return g4f_openai_call(model2, system_content, user_content+'\n'+srt_text)
        else:
            logging.error("LLM name error, only {} are supported as LLM name prefix."
                          .format(SUPPORT_LLM_PREFIX))

    def AI_clip(LLM_res, dest_text, video_spk_input, start_ost, end_ost, video_state, audio_state, output_dir):
        timestamp_list = extract_timestamps(LLM_res)
        output_dir = output_dir.strip()
        output_dir = os.path.abspath(output_dir) if output_dir else None
        if video_state is not None:
            clip_video_file, message, clip_srt = audio_clipper.video_clip(
                dest_text, start_ost, end_ost, video_state,
                dest_spk=video_spk_input, output_dir=output_dir, timestamp_list=timestamp_list, add_sub=False)
            return clip_video_file, None, message, clip_srt
        if audio_state is not None:
            (sr, res_audio), message, clip_srt = audio_clipper.clip(
                dest_text, start_ost, end_ost, audio_state,
                dest_spk=video_spk_input, output_dir=output_dir, timestamp_list=timestamp_list, add_sub=False)
            return None, (sr, res_audio), message, clip_srt

    def AI_clip_subti(LLM_res, dest_text, video_spk_input, start_ost, end_ost, video_state, audio_state, output_dir):
        timestamp_list = extract_timestamps(LLM_res)
        output_dir = output_dir.strip()
        output_dir = os.path.abspath(output_dir) if output_dir else None
        if video_state is not None:
            clip_video_file, message, clip_srt = audio_clipper.video_clip(
                dest_text, start_ost, end_ost, video_state,
                dest_spk=video_spk_input, output_dir=output_dir, timestamp_list=timestamp_list, add_sub=True)
            return clip_video_file, None, message, clip_srt
        if audio_state is not None:
            (sr, res_audio), message, clip_srt = audio_clipper.clip(
                dest_text, start_ost, end_ost, audio_state,
                dest_spk=video_spk_input, output_dir=output_dir, timestamp_list=timestamp_list, add_sub=True)
            return None, (sr, res_audio), message, clip_srt

    # ============= 新增：语义分析触发 & 用高光一键剪辑 =============
    def semantic_analyze_action(srt_text, apikey, model):
        """按钮：执行语义摘要/关键词/大纲/高光识别，返回 JSON 字符串"""
        return semantic_analysis_run(srt_text, apikey, model)

    def semantic_clip_action(analysis_json_str, video_text_input, video_spk_input,
                             start_ost, end_ost, video_state, audio_state, output_dir):
        """按钮：把语义分析中的高光 highlights 直接转成剪辑（不烧录字幕）"""
        ts_list = highlights_to_timestamps(analysis_json_str)  # [[s, e], ...] 秒

        # ✅ 关键修复：为兼容底层旧实现，这里把“秒”预先乘以 1000 作为“毫秒”传下去
        ts_list_scaled = _compat_scale_seconds_for_legacy(ts_list)

        output_dir = (output_dir or "").strip()
        output_dir = os.path.abspath(output_dir) if output_dir else None

        ranges_pretty = ", ".join([f"[{round(s,2)}-{round(e,2)}]" for s, e in ts_list]) if ts_list else "(无)"

        if video_state is not None:
            clip_video_file, message, clip_srt = audio_clipper.video_clip(
                video_text_input, start_ost, end_ost, video_state,
                dest_spk=video_spk_input, output_dir=output_dir, timestamp_list=ts_list_scaled, add_sub=False)
            message = f"{message}\n(按语义高光剪辑: {ranges_pretty})"
            return clip_video_file, None, message, clip_srt
        if audio_state is not None:
            (sr, res_audio), message, clip_srt = audio_clipper.clip(
                video_text_input, start_ost, end_ost, audio_state,
                dest_spk=video_spk_input, output_dir=output_dir, timestamp_list=ts_list_scaled, add_sub=False)
            message = f"{message}\n(按语义高光剪音频: {ranges_pretty})"
            return None, (sr, res_audio), message, clip_srt
        return None, None, "未发现可用的识别状态（video_state/audio_state 均为空）。请先进行识别。", ""
    # ===============================================================

    # ======================
    # ===== Gradio UI  =====
    # ======================
    theme = gr.Theme.load("funclip/utils/theme_2.json")
    with gr.Blocks(theme=theme) as funclip_service:
        gr.Markdown(top_md_1)

        video_state, audio_state = gr.State(), gr.State()

        with gr.Row():
            # ===== 左侧：输入与识别 =====
            with gr.Column():
                with gr.Row():
                    video_input = gr.Video(label="视频输入 | Video Input")
                    audio_input = gr.Audio(label="音频输入 | Audio Input")

                with gr.Column():
                    hotwords_input = gr.Textbox(label="🚒 热词 | Hotwords(可以为空，仅支持中文)")
                    output_dir = gr.Textbox(label="📁 文件输出路径 | File Output Dir (可以为空)", value=" ")
                    with gr.Row():
                        recog_button = gr.Button("👂 识别 | ASR", variant="primary")
                        recog_button2 = gr.Button("👂👫 识别+区分说话人 | ASR+SD")

                # 展示空间加大（lines=16）
                video_text_output = gr.Textbox(label="✏️ 识别结果 | Recognition Result", lines=16, scale=1)
                video_srt_output = gr.Textbox(label="📖 SRT字幕内容 | RST Subtitles", lines=16, scale=1)

            # ===== 右侧：LLM 智能剪 / 文本剪 / 语义分析 =====
            with gr.Column():
                with gr.Tab("🧠 LLM智能裁剪 | LLM Clipping"):
                    with gr.Column():
                        prompt_head = gr.Textbox(
                            label="Prompt System",
                            value=("你是一个专业的视频字幕分析与剪辑助手。输入内容是视频的完整 SRT 字幕文本，请你完成以下任务：\n"
                                   "1. 从中选取最有信息量、语义完整且连续的片段；\n"
                                   "2. 对时间上连续的多个句子进行合并，确保文字与时间戳一一对应；\n"
                                   "3. 优先选择表达核心观点、情感高潮或主题转折的部分；\n"
                                   "4. 根据视频的总时长或信息密度动态调整片段数量。"
                                   "5. 严格按照以下格式输出，每条独占一行：\n"
                                   "    1. [开始时间-结束时间] 内容文本\n"
                                   "    2. [开始时间-结束时间] 内容文本\n"
                                   "⚠️ 仅输出结果；使用半角“-”；\n" 
                                   "以下是视频的语义理解与摘要供你参考：")
                        )
                        prompt_head2 = gr.Textbox(label="Prompt User", value=("这是待裁剪的视频srt字幕："))
                        with gr.Column():
                            with gr.Row():
                                llm_model = gr.Dropdown(
                                    choices=[
                                        "deepseek-chat",
                                        "qwen-plus",
                                        "gpt-3.5-turbo",
                                        "gpt-3.5-turbo-0125",
                                        "gpt-4-turbo",
                                        "g4f-gpt-3.5-turbo"
                                    ],
                                    value="deepseek-chat",
                                    label="LLM Model Name",
                                    allow_custom_value=True
                                )
                                apikey_input = gr.Textbox(label="APIKEY")
                            llm_button = gr.Button("LLM推理 | LLM Inference", variant="primary")
                        llm_result = gr.Textbox(label="LLM Clipper Result", lines=14, scale=1)
                        with gr.Row():
                            llm_clip_button = gr.Button("🧠 LLM智能裁剪 | AI Clip", variant="primary")
                            llm_clip_subti_button = gr.Button("🧠 LLM智能裁剪+字幕 | AI Clip+Subtitles")

                with gr.Tab("✂️ 根据文本/说话人裁剪 | Text/Speaker Clipping"):
                    video_text_input = gr.Textbox(label="✏️ 待裁剪文本 | Text to Clip (多段文本使用'#'连接)")
                    video_spk_input = gr.Textbox(label="✏️ 待裁剪说话人 | Speaker to Clip (多个说话人使用'#'连接)")
                    with gr.Row():
                        clip_button = gr.Button("✂️ 裁剪 | Clip", variant="primary")
                        clip_subti_button = gr.Button("✂️ 裁剪+字幕 | Clip+Subtitles")
                    with gr.Row():
                        video_start_ost = gr.Slider(minimum=-500, maximum=1000, value=0, step=50,
                                                    label="⏪ 开始位置偏移 | Start Offset (ms)")
                        video_end_ost = gr.Slider(minimum=-500, maximum=1000, value=100, step=50,
                                                  label="⏩ 结束位置偏移 | End Offset (ms)")

                # ===== 新增 Tab：语义理解 / 摘要 / 高光提取 =====
                with gr.Tab("🧩 语义理解与摘要 | Semantic Analysis"):
                    with gr.Row():
                        sa_model = gr.Dropdown(
                            choices=[
                                "deepseek-chat",
                                "qwen-plus",
                                "gpt-3.5-turbo",
                                "gpt-3.5-turbo-0125",
                                "gpt-4-turbo",
                                "g4f-gpt-3.5-turbo"
                            ],
                            value="qwen-plus",
                            label="LLM Model Name (Semantic)"
                        )
                        sa_apikey = gr.Textbox(label="APIKEY (Semantic)")
                        sa_button = gr.Button("🔎 语义摘要/关键词/大纲/高光", variant="primary")
                    sa_result = gr.Textbox(label="语义分析结果（JSON）", lines=18, scale=1)
                    with gr.Row():
                        sa_clip_button = gr.Button("✨ 用高光一键剪辑（不加字幕）", variant="primary")

                with gr.Row():
                    font_size = gr.Slider(minimum=10, maximum=100, value=32, step=2,
                                          label="🔠 字幕字体大小 | Subtitle Font Size")
                    font_color = gr.Radio(["black", "white", "green", "red"],
                                          label="🌈 字幕颜色 | Subtitle Color", value='white')

                video_output = gr.Video(label="裁剪结果 | Video Clipped")
                audio_output = gr.Audio(label="裁剪结果 | Audio Clipped")
                clip_message = gr.Textbox(label="⚠️ 裁剪信息 | Clipping Log", lines=10, scale=1)
                srt_clipped = gr.Textbox(label="📖 裁剪部分SRT字幕内容 | Clipped RST Subtitles", lines=10, scale=1)

        # ===== 事件绑定 =====
        recog_button.click(
            mix_recog,
            inputs=[video_input, audio_input, hotwords_input, output_dir],
            outputs=[video_text_output, video_srt_output, video_state, audio_state]
        )
        recog_button2.click(
            mix_recog_speaker,
            inputs=[video_input, audio_input, hotwords_input, output_dir],
            outputs=[video_text_output, video_srt_output, video_state, audio_state]
        )
        clip_button.click(
            mix_clip,
            inputs=[video_text_input, video_spk_input, video_start_ost, video_end_ost, video_state, audio_state, output_dir],
            outputs=[video_output, audio_output, clip_message, srt_clipped]
        )
        clip_subti_button.click(
            video_clip_addsub,
            inputs=[video_text_input, video_spk_input, video_start_ost, video_end_ost, video_state, output_dir, font_size, font_color],
            outputs=[video_output, clip_message, srt_clipped]
        )
        llm_button.click(
            llm_inference,
            inputs=[prompt_head, prompt_head2, video_srt_output, llm_model, apikey_input],
            outputs=[llm_result]
        )
        llm_clip_button.click(
            AI_clip,
            inputs=[llm_result, video_text_input, video_spk_input, video_start_ost, video_end_ost, video_state, audio_state, output_dir],
            outputs=[video_output, audio_output, clip_message, srt_clipped]
        )
        llm_clip_subti_button.click(
            AI_clip_subti,
            inputs=[llm_result, video_text_input, video_spk_input, video_start_ost, video_end_ost, video_state, audio_state, output_dir],
            outputs=[video_output, audio_output, clip_message, srt_clipped]
        )

        # 新增绑定：语义分析 & 用高光一键剪辑
        sa_button.click(
            semantic_analyze_action,
            inputs=[video_srt_output, sa_apikey, sa_model],
            outputs=[sa_result]
        )
        sa_clip_button.click(
            semantic_clip_action,
            inputs=[sa_result, video_text_input, video_spk_input, video_start_ost, video_end_ost, video_state, audio_state, output_dir],
            outputs=[video_output, audio_output, clip_message, srt_clipped]
        )

    # ===== 启动服务 =====
    if args.listen:
        funclip_service.launch(share=args.share, server_port=args.port, server_name=server_name, inbrowser=False)
    else:
        funclip_service.launch(share=args.share, server_port=args.port, server_name=server_name)



