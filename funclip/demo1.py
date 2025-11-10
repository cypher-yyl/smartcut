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

def _srt_time_to_seconds(t: str) -> float:
    """'HH:MM:SS,ms' 或 'MM:SS,ms' → 秒(float)；纯数字则尝试直接转 float。"""
    if not isinstance(t, str):
        return float(t)
    s = t.strip().replace(".", ",")
    m = re.match(r"^(\d{2}):(\d{2}):(\d{2}),(\d{1,3})$", s)
    if m:
        hh, mm, ss, ms = m.groups()
        return int(hh)*3600 + int(mm)*60 + int(ss) + int(ms)/1000.0
    m2 = re.match(r"^(\d{2}):(\d{2}),(\d{1,3})$", s)
    if m2:
        mm, ss, ms = m2.groups()
        return int(mm)*60 + int(ss) + int(ms)/1000.0
    return float(s)  # 兜底：纯秒/纯毫秒/可解析数字

def _coerce_ts_to_seconds(ts_list):
    """把 [[s,e], ...]（可能是SRT字符串/毫秒/秒）统一成 秒(float)。自动识别>600视为毫秒/或SRT已转秒。"""
    out = []
    for pair in ts_list or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        s, e = pair
        # 字符串 → 秒
        if isinstance(s, str): s = _srt_time_to_seconds(s)
        if isinstance(e, str): e = _srt_time_to_seconds(e)
        # 明显像毫秒（经验阈：>600 且 < 1e7）→ 换成秒
        if isinstance(s, (int, float)) and isinstance(e, (int, float)):
            if (s > 600 or e > 600) and (s < 1e7 and e < 1e7):
                s, e = s/1000.0, e/1000.0
        try:
            s = float(s); e = float(e)
            if e > s:  # 长度过滤后面再做
                out.append([round(s, 3), round(e, 3)])
        except Exception:
            continue
    return out

def _filter_and_clip_to_duration(ts_sec, duration_sec: float, min_len=2.0):
    """裁剪越界并过滤过短区间（单位：秒）。"""
    ok = []
    if not duration_sec or duration_sec <= 0:
        duration_sec = 9e9  # 没拿到时长就先放行
    for s, e in ts_sec:
        s2 = max(0.0, min(float(s), duration_sec))
        e2 = max(0.0, min(float(e), duration_sec))
        if e2 > s2 and (e2 - s2) >= min_len:
            ok.append([round(s2, 3), round(e2, 3)])
    return ok

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

# ======== 新增：从语义 JSON 组装上下文，用于提示词增强 ========
def build_semantic_context(analysis_json_str: str, max_chars: int = 900) -> str:
    """将语义分析 JSON 压缩成可控长度的上下文提示；容错各种奇怪结构。"""
    try:
        data = json.loads(analysis_json_str) if isinstance(analysis_json_str, str) and analysis_json_str.strip() else {}
    except Exception:
        return ""

    if not isinstance(data, dict):
        return ""

    def _to_list(val, n=10):
        if isinstance(val, list):
            out = []
            for x in val[:n]:
                if isinstance(x, (str, int, float)):
                    out.append(str(x))
                elif isinstance(x, dict):
                    # 从常见字段兜底取一个文本
                    for k in ("text", "name", "title", "phrase", "kw", "keyword", "desc", "reason"):
                        if k in x and x[k]:
                            out.append(str(x[k]))
                            break
                else:
                    out.append(str(x))
            return out
        return []

    topics = _to_list(data.get("topics"), n=5)
    kws    = _to_list(data.get("keywords"), n=30)

    raw_notes = data.get("notes", "")
    if isinstance(raw_notes, str):
        notes = raw_notes.strip()
    elif isinstance(raw_notes, list):
        notes = " | ".join(_to_list(raw_notes, n=8))
    elif isinstance(raw_notes, dict):
        # 把 dict 压成一行
        try:
            notes = json.dumps(raw_notes, ensure_ascii=False)
        except Exception:
            notes = str(raw_notes)
    else:
        notes = str(raw_notes) if raw_notes is not None else ""

    ents = data.get("entities", {}) if isinstance(data.get("entities"), dict) else {}
    persons = _to_list(ents.get("person"), n=10)
    orgs    = _to_list(ents.get("org"), n=10)
    prods   = _to_list(ents.get("product"), n=10)
    techs   = _to_list(ents.get("tech_term"), n=10)

    outline_lines = []
    if isinstance(data.get("outline"), list):
        for o in data["outline"][:8]:
            if isinstance(o, dict):
                title = (o.get("title") or o.get("chapter") or o.get("name") or "").strip()
                s = o.get("start") or o.get("begin") or o.get("from")
                e = o.get("end") or o.get("to")
                if title:
                    outline_lines.append(f"- {title} ({s}-{e}s)" if s is not None and e is not None else f"- {title}")
            elif isinstance(o, str) and o.strip():
                outline_lines.append(f"- {o.strip()}")

    ctx = [
        "### Semantic Context (for clipping)",
        f"Topics: {', '.join(topics)}" if topics else "",
        f"Keywords: {', '.join(kws)}" if kws else "",
        f"Notes: {notes}" if notes else "",
        "Entities:",
        f"  - person: {', '.join(persons)}" if persons else "",
        f"  - org: {', '.join(orgs)}" if orgs else "",
        f"  - product: {', '.join(prods)}" if prods else "",
        f"  - tech_term: {', '.join(techs)}" if techs else "",
        "Outline:",
        *outline_lines
    ]
    text = "\n".join([c for c in ctx if c]).strip()
    return (text[:max_chars] + "…") if len(text) > max_chars else text


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

# ============= 新增：时间片段合并去重（秒） =============
def _merge_timestamp_ranges(ts_a, ts_b, max_segments=6, min_len=2.0):
    """
    ts_a/ts_b: [[s, e], ...]（秒）
    - 合并后做排序、去重（允许小重叠时合并）
    - 过滤短片段
    """
    all_ts = []
    for pair in (ts_a or []):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            all_ts.append([float(pair[0]), float(pair[1])])
    for pair in (ts_b or []):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            all_ts.append([float(pair[0]), float(pair[1])])
    # 过滤异常
    all_ts = [p for p in all_ts if p[1] > p[0] and (p[1] - p[0]) >= min_len]
    if not all_ts:
        return []
    # 排序并对小重叠/相邻段进行合并（阈值：0.6s）
    all_ts.sort(key=lambda x: (x[0], x[1]))
    merged = []
    EPS = 0.6
    for s, e in all_ts:
        if not merged or s > merged[-1][1] + EPS:
            merged.append([round(s, 2), round(e, 2)])
        else:
            merged[-1][1] = round(max(merged[-1][1], e), 2)
    return merged[:max_segments]

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

    # ======== 改造：支持“语义上下文增强”的 LLM 推理 ========
    def llm_inference(system_content, user_content, srt_text, model, apikey,
                      use_semantic_context=False, semantic_json_str=""):
        """
        兼容旧用法：后两个参数有默认值。前端若不传，逻辑与原版一致。
        当 use_semantic_context=True 时，会把 build_semantic_context(semantic_json_str)
        以“补充说明”拼接到 user_content 之前，帮助模型选段。
        """
        SUPPORT_LLM_PREFIX = ['qwen', 'gpt', 'g4f', 'moonshot', 'deepseek']
        # 构造上下文
        extra_ctx = ""
        if use_semantic_context and semantic_json_str:
            extra_ctx = build_semantic_context(semantic_json_str)
            if extra_ctx:
                # 将上下文放到用户提示前（避免破坏 system 的角色）
                user_content = f"[Semantic Context]\n{extra_ctx}\n\n[Task]\n{user_content}"

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

    # ======== 改造：AI 剪辑支持语义高光回退 / 合并 ========
    def AI_clip(LLM_res, dest_text, video_spk_input, start_ost, end_ost,
                video_state, audio_state, output_dir,
                use_semantic_context=False, semantic_json_str=""):

        # 1) LLM 提取 + 语义高光 → 统一到“秒”
        raw_llm = extract_timestamps(LLM_res) or []
        ts_from_llm = _coerce_ts_to_seconds(raw_llm)

        ts_from_sa = []
        if use_semantic_context and semantic_json_str:
            ts_from_sa = _coerce_ts_to_seconds(highlights_to_timestamps(semantic_json_str))

        ts_merged = _merge_timestamp_ranges(ts_from_llm, ts_from_sa, max_segments=6, min_len=2.0)

        # 2) 获取素材时长（可选：从 state 上取；没有就不截断）
        duration_sec = None
        try:
            media_state = video_state if video_state is not None else audio_state
            duration_sec = float(getattr(media_state, "duration", None)) if media_state is not None else None
        except Exception:
            duration_sec = None

        ts_final = _filter_and_clip_to_duration(ts_merged, duration_sec or 0.0, min_len=2.0)

        output_dir = (output_dir or "").strip()
        output_dir = os.path.abspath(output_dir) if output_dir else None
        ranges_pretty = ", ".join([f"[{s}-{e}]" for s, e in ts_final]) if ts_final else "(无)"

        # 3) 直接把“秒”传给底层（moviepy 就是秒）
        if video_state is not None:
            clip_video_file, message, clip_srt = audio_clipper.video_clip(
                dest_text, start_ost, end_ost, video_state,
                dest_spk=video_spk_input, output_dir=output_dir,
                timestamp_list=ts_final, add_sub=False)
            suffix = "\n(启用语义上下文)" if (use_semantic_context and semantic_json_str) else ""
            message = f"{message}\n(按 LLM/语义高光合并剪辑: {ranges_pretty}){suffix}"
            return clip_video_file, None, message, clip_srt

        if audio_state is not None:
            (sr, res_audio), message, clip_srt = audio_clipper.clip(
                dest_text, start_ost, end_ost, audio_state,
                dest_spk=video_spk_input, output_dir=output_dir,
                timestamp_list=ts_final, add_sub=False)
            suffix = "\n(启用语义上下文)" if (use_semantic_context and semantic_json_str) else ""
            message = f"{message}\n(按 LLM/语义高光合并剪音频: {ranges_pretty}){suffix}"
            return None, (sr, res_audio), message, clip_srt


    
    def AI_clip_subti(LLM_res, dest_text, video_spk_input, start_ost, end_ost,
                    video_state, audio_state, output_dir,
                    use_semantic_context=False, semantic_json_str=""):

        raw_llm = extract_timestamps(LLM_res) or []
        ts_from_llm = _coerce_ts_to_seconds(raw_llm)

        ts_from_sa = []
        if use_semantic_context and semantic_json_str:
            ts_from_sa = _coerce_ts_to_seconds(highlights_to_timestamps(semantic_json_str))

        ts_merged = _merge_timestamp_ranges(ts_from_llm, ts_from_sa, max_segments=6, min_len=2.0)

        duration_sec = None
        try:
            media_state = video_state if video_state is not None else audio_state
            duration_sec = float(getattr(media_state, "duration", None)) if media_state is not None else None
        except Exception:
            duration_sec = None

        ts_final = _filter_and_clip_to_duration(ts_merged, duration_sec or 0.0, min_len=2.0)

        output_dir = (output_dir or "").strip()
        output_dir = os.path.abspath(output_dir) if output_dir else None
        ranges_pretty = ", ".join([f"[{s}-{e}]" for s, e in ts_final]) if ts_final else "(无)"

        if video_state is not None:
            clip_video_file, message, clip_srt = audio_clipper.video_clip(
                dest_text, start_ost, end_ost, video_state,
                dest_spk=video_spk_input, output_dir=output_dir,
                timestamp_list=ts_final, add_sub=True)
            suffix = "\n(启用语义上下文)" if (use_semantic_context and semantic_json_str) else ""
            message = f"{message}\n(按 LLM/语义高光合并剪辑+字幕: {ranges_pretty}){suffix}"
            return clip_video_file, None, message, clip_srt

        if audio_state is not None:
            (sr, res_audio), message, clip_srt = audio_clipper.clip(
                dest_text, start_ost, end_ost, audio_state,
                dest_spk=video_spk_input, output_dir=output_dir,
                timestamp_list=ts_final, add_sub=True)
            suffix = "\n(启用语义上下文)" if (use_semantic_context and semantic_json_str) else ""
            message = f"{message}\n(按 LLM/语义高光合并剪音频+字幕: {ranges_pretty}){suffix}"
            return None, (sr, res_audio), message, clip_srt


    # ============= 新增：语义分析触发 & 用高光一键剪辑 =============
    def semantic_analyze_action(srt_text, apikey, model):
        """按钮：执行语义摘要/关键词/大纲/高光识别，返回 JSON 字符串"""
        return semantic_analysis_run(srt_text, apikey, model)

    def semantic_clip_action(analysis_json_str, video_text_input, video_spk_input,
                            start_ost, end_ost, video_state, audio_state, output_dir):
        ts_list = _coerce_ts_to_seconds(highlights_to_timestamps(analysis_json_str))

        duration_sec = None
        try:
            media_state = video_state if video_state is not None else audio_state
            duration_sec = float(getattr(media_state, "duration", None)) if media_state is not None else None
        except Exception:
            duration_sec = None

        ts_final = _filter_and_clip_to_duration(ts_list, duration_sec or 0.0, min_len=2.0)

        output_dir = (output_dir or "").strip()
        output_dir = os.path.abspath(output_dir) if output_dir else None
        ranges_pretty = ", ".join([f"[{s}-{e}]" for s, e in ts_final]) if ts_final else "(无)"

        if video_state is not None:
            clip_video_file, message, clip_srt = audio_clipper.video_clip(
                video_text_input, start_ost, end_ost, video_state,
                dest_spk=video_spk_input, output_dir=output_dir,
                timestamp_list=ts_final, add_sub=False)
            message = f"{message}\n(按语义高光剪辑: {ranges_pretty})"
            return clip_video_file, None, message, clip_srt

        if audio_state is not None:
            (sr, res_audio), message, clip_srt = audio_clipper.clip(
                video_text_input, start_ost, end_ost, audio_state,
                dest_spk=video_spk_input, output_dir=output_dir,
                timestamp_list=ts_final, add_sub=False)
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
                                   "4. 严格按照以下格式输出，每条独占一行：\n"
                                   "1. [开始时间-结束时间] 内容文本\n"
                                   "2. [开始时间-结束时间] 内容文本\n"
                                   "⚠️ 仅输出结果；使用半角“-”；")
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
                            # ==== 新增：是否启用语义上下文增强 ====
                            use_sa_ckb = gr.Checkbox(value=True, label="🔗 使用语义上下文增强（来自『语义理解与摘要』Tab）")
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
        # ==== 改造：把“是否使用语义上下文 + 语义 JSON” 也传入 ====
        llm_button.click(
            llm_inference,
            inputs=[prompt_head, prompt_head2, video_srt_output, llm_model, apikey_input, use_sa_ckb, sa_result],
            outputs=[llm_result]
        )
        llm_clip_button.click(
            AI_clip,
            inputs=[llm_result, video_text_input, video_spk_input, video_start_ost, video_end_ost,
                    video_state, audio_state, output_dir, use_sa_ckb, sa_result],
            outputs=[video_output, audio_output, clip_message, srt_clipped]
        )
        llm_clip_subti_button.click(
            AI_clip_subti,
            inputs=[llm_result, video_text_input, video_spk_input, video_start_ost, video_end_ost,
                    video_state, audio_state, output_dir, use_sa_ckb, sa_result],
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
