import os
import re
import tempfile
import urllib.parse
from typing import Optional, Tuple

# 可选依赖：yt_dlp（优先使用，失败回退到 requests 直链）
try:
    import yt_dlp  # type: ignore
except Exception:
    yt_dlp = None


def _ensure_output_dir(output_dir: Optional[str]) -> str:
    """确保输出目录存在；为空则使用临时目录。"""
    if output_dir:
        out = os.path.abspath(output_dir.strip())
        os.makedirs(out, exist_ok=True)
        return out
    return tempfile.mkdtemp(prefix="funclip_dl_")


def _guess_filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = os.path.basename(parsed.path) or "downloaded_video"
    if "." not in name:
        name += ".mp4"
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name


def _download_with_requests(url: str, dst_path: str) -> str:
    import requests
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dst_path


def download_video_by_url(url: str, output_dir: Optional[str]) -> Tuple[Optional[str], str]:
    """
    下载视频（优先 yt_dlp，失败回落到 requests 直链）。
    返回：(本地文件路径 or None, 日志信息)
    """
    url = (url or "").strip()
    if not url:
        return None, "❌ URL 为空。"

    outdir = _ensure_output_dir(output_dir)
    log_lines = [f"➡️ 开始下载：{url}", f"📁 输出目录：{outdir}"]

    # 1) 优先 yt_dlp：适配 YouTube/B站/抖音等
    if yt_dlp is not None:
        ydl_opts = {
            "outtmpl": os.path.join(outdir, "%(title).200B.%(ext)s"),
            "ignoreerrors": True,
            "noprogress": True,
            "quiet": True,
            "merge_output_format": "mp4",
            "format": "bv*+ba/b",
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    return None, "\n".join(log_lines + ["❌ yt_dlp 返回空信息"])
                if "entries" in info:
                    for ent in info["entries"] or []:
                        if ent and ent.get("_filename"):
                            fp = ent["_filename"]
                            log_lines.append(f"✅ 下载完成：{fp}")
                            return fp, "\n".join(log_lines)
                    return None, "\n".join(log_lines + ["❌ 未取得可用的条目文件名"])
                else:
                    fp = info.get("_filename")
                    if fp:
                        log_lines.append(f"✅ 下载完成：{fp}")
                        return fp, "\n".join(log_lines)
                    return None, "\n".join(log_lines + ["❌ 未取得文件名"])
        except Exception as e:
            log_lines.append(f"⚠️ yt_dlp 下载失败：{e}，尝试直链下载…")

    # 2) 回退：requests 直链
    try:
        filename = _guess_filename_from_url(url)
        dst = os.path.join(outdir, filename)
        _download_with_requests(url, dst)
        log_lines.append(f"✅ 下载完成（直链）：{dst}")
        return dst, "\n".join(log_lines)
    except Exception as e:
        log_lines.append(f"❌ 直链下载失败：{e}")
        return None, "\n".join(log_lines)


def download_video_action(url: str, output_dir: str):
    """
    Gradio 回调：输入 URL、输出目录
    输出：
      - downloaded_video_preview: 供 Video 组件预览
      - downloaded_file_path: 存储到文本框，后续识别优先使用
      - message: 合并到“裁剪信息 | Clipping Log”
    """
    output_dir = (output_dir or "").strip() or None
    fp, log_msg = download_video_by_url(url, output_dir)
    if fp and os.path.exists(fp):
        return fp, fp, f"[URL下载]\n{log_msg}"
    else:
        return None, "", f"[URL下载]\n{log_msg}"


def resolve_video_input(video_input, downloaded_path: Optional[str]):
    """优先使用已下载文件路径；否则使用上传的视频输入。"""
    if downloaded_path:
        p = downloaded_path.strip()
        if p and os.path.exists(p):
            return p
    return video_input
