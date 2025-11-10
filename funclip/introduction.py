top_md_1 = ("""
<div align="center" style="margin-top:10px;">
  <h2 style="margin:6px 0;">🎬 SmartCut 智能视频剪辑系统</h2>
  <p style="font-size:14px;opacity:.85;">项目小组：<b>第五组</b></p>
</div>
<ul style="margin:6px 0;">
  <li>① 上传视频或音频并点击 <b><font color="#f7802b">识别</font></b></li>
  <li>② 在右侧选择文本或说话人裁剪，或使用 <b>LLM 智能剪辑</b></li>
  <li>③ 点击 <b><font color="#f7802b">裁剪</font></b> / <b><font color="#f7802b">裁剪+字幕</font></b> 生成结果</li>
</ul>
""")


top_md_3 = ("""访问FunASR项目与论文能够帮助您深入了解ParaClipper中所使用的语音处理相关模型：
    <div align="center">
    <div style="display:flex; gap: 0.25rem;" align="center">
        FunASR: <a href='https://github.com/alibaba-damo-academy/FunASR'><img src='https://img.shields.io/badge/Github-Code-blue'></a> 
        FunASR Paper: <a href="https://arxiv.org/abs/2305.11013"><img src="https://img.shields.io/badge/Arxiv-2305.11013-orange"></a> 
        🌟Star FunASR: <a href='https://github.com/alibaba-damo-academy/FunASR/stargazers'><img src='https://img.shields.io/github/stars/alibaba-damo-academy/FunASR.svg?style=social'></a>
    </div>
    </div>
    """)

top_md_4 = ("""我们在「LLM智能裁剪」模块中提供三种LLM调用方式，
            1. 选择阿里云百炼平台通过api调用qwen系列模型，此时需要您准备百炼平台的apikey，请访问[阿里云百炼](https://bailian.console.aliyun.com/#/home)；
            2. 选择GPT开头的模型即为调用openai官方api，此时需要您自备sk与网络环境；
            3. [gpt4free](https://github.com/xtekky/gpt4free?tab=readme-ov-file)项目也被集成进FunClip，可以通过它免费调用gpt模型；
            
            其中方式1与方式2需要在界面中传入相应的apikey        
            方式3而可能非常不稳定，返回时间可能很长或者结果获取失败，可以多多尝试或者自己准备sk使用方式1,2
            
            不要同时打开同一端口的多个界面，会导致文件上传非常缓慢或卡死，关闭其他界面即可解决
            """)
