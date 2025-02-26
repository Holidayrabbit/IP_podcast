import os
import json
import time
from typing import List, Dict, Any
from pathlib import Path
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key
from elevenlabs import Voice
import json


# 加载环境变量
load_dotenv()

# 设置 ElevenLabs API 密钥
set_api_key(os.getenv("ELEVENLABS_API_KEY"))

# 设置 OpenRouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 定义模型
DEEPSEEK_MODEL = "deepseek-ai/deepseek-chat-r1"
GEMINI_MODEL = "google/gemini-pro-2.0-experimental"

# 创建 LangChain 模型实例
deepseek_model = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

gemini_model = ChatOpenAI(
    model=GEMINI_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# 定义 ElevenLabs 声音
VOICE_A = "Adam"  # 替换为您想要的声音ID
VOICE_B = "Rachel"  # 替换为您想要的声音ID

def read_text_file(file_path: str) -> str:
    """读取文本文件内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def generate_core_topics(podcast_theme: str, book_summary: str, duration_minutes: int) -> str:
    """使用DeepSeek生成核心话题"""
    prompt_template = """
    你是一位专业的播客内容策划师。请根据以下播客主题、书籍摘要和预期对话时长，生成适合双人对话播客的核心话题列表。
    如果对话时长较短比如5分钟以内，提供的话题数量为3个。否则提供5个。
    这些核心话题将用于指导播客对话的方向和深度。

    播客主题:
    {podcast_theme}

    书籍摘要:
    {book_summary}

    预期对话时长: {duration_minutes}分钟

    每个话题应该包含:
    1. 话题标题
    2. 话题描述

    输出的json格式如下：
    {
        "core_topics": [
            {
                "topic_title": "话题标题",
                "topic_description": "话题描述"
            }
        ]
    }

    确保话题之间有逻辑连贯性。
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | deepseek_model | StrOutputParser()
    
    result = chain.invoke({
        "book_summary": book_summary,
        "podcast_theme": podcast_theme,
        "duration_minutes": duration_minutes
    })
    
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    
    # 尝试解析JSON并保存
    try:
        # 解析JSON字符串
        topics_data = json.loads(result)
        # 保存到文件
        with open("output/core_topics.json", "w", encoding="utf-8") as f:
            json.dump(topics_data, f, ensure_ascii=False, indent=4)
    except json.JSONDecodeError:
        # 如果不是有效的JSON，则直接保存文本
        with open("output/core_topics.txt", "w", encoding="utf-8") as f:
            f.write(result)
    
    return result

def generate_podcast_transcript(book_summary: str, core_topics: str) -> str:
    """使用Gemini生成播客对话脚本"""
    prompt_template = """
    你是一位专业的播客脚本撰写者。请根据以下书籍摘要和核心话题，创建一段自然、有趣且信息丰富的双人对话播客脚本。

    书籍摘要:
    {book_summary}

    核心话题:
    {core_topics}

    请按照以下格式创建对话脚本:

    ****** opening ******
    A [兴奋]: (开场白)
    B [友好]: (回应)

    ****** content ******
    A [情感1]: (讨论第一个话题)
    B [情感2]: (回应并深入讨论)
    A [情感3]: (提出问题或新观点)
    B [情感4]: (回应)
    ...

    ****** content ******
    (继续下一个话题的讨论)
    ...

    ****** closing ******
    A [满足]: (总结讨论)
    B [感激]: (结束语)

    要求:
    1. 对话应该自然流畅，像真实的人在交谈
    2. 每个发言者的语气和情感应在方括号中标注
    3. 包含开场白、多个内容部分和结束语
    4. 确保内容覆盖所有核心话题
    5. 避免过长的独白，保持互动性
    6. 总对话长度应适合{duration_minutes}分钟的播客
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | gemini_model | StrOutputParser()
    
    return chain.invoke({
        "book_summary": book_summary,
        "core_topics": core_topics,
        "duration_minutes": calculate_duration_from_topics(core_topics)
    })

def calculate_duration_from_topics(core_topics: str) -> int:
    """从核心话题估算总时长"""
    # 简单实现，实际应用中可能需要更复杂的解析
    try:
        # 尝试找出所有包含"分钟"的行并提取数字
        import re
        minutes = re.findall(r'(\d+)\s*分钟', core_topics)
        if minutes:
            return sum(map(int, minutes))
        return 20  # 默认20分钟
    except:
        return 20  # 默认20分钟

def parse_transcript(transcript: str) -> List[Dict[str, Any]]:
    """解析生成的脚本为结构化数据"""
    segments = []
    lines = transcript.strip().split('\n')
    
    current_speaker = None
    current_emotion = None
    current_text = ""
    
    for line in lines:
        line = line.strip()
        if not line or "******" in line:
            # 保存当前段落并重置
            if current_speaker and current_text:
                segments.append({
                    "speaker": current_speaker,
                    "emotion": current_emotion,
                    "text": current_text.strip()
                })
                current_text = ""
            continue
            
        # 尝试解析说话者和情感
        import re
        match = re.match(r'([AB])\s*\[(.*?)\]:\s*(.*)', line)
        if match:
            # 保存之前的段落
            if current_speaker and current_text:
                segments.append({
                    "speaker": current_speaker,
                    "emotion": current_emotion,
                    "text": current_text.strip()
                })
            
            current_speaker = match.group(1)
            current_emotion = match.group(2)
            current_text = match.group(3)
        else:
            # 继续当前段落
            current_text += " " + line
    
    # 添加最后一个段落
    if current_speaker and current_text:
        segments.append({
            "speaker": current_speaker,
            "emotion": current_emotion,
            "text": current_text.strip()
        })
    
    return segments

async def generate_audio_segment(segment: Dict[str, Any], output_dir: Path) -> str:
    """为单个对话段落生成音频"""
    speaker = segment["speaker"]
    text = segment["text"]
    emotion = segment["emotion"]
    
    # 选择声音
    voice = VOICE_A if speaker == "A" else VOICE_B
    
    # 根据情感调整语音参数
    stability = 0.5
    similarity_boost = 0.5
    
    if emotion.lower() in ["兴奋", "激动", "热情"]:
        stability = 0.3
    elif emotion.lower() in ["平静", "思考", "严肃"]:
        stability = 0.7
    
    # 生成音频
    audio = generate(
        text=text,
        voice=voice,
        model="eleven_multilingual_v2",
        stability=stability,
        similarity_boost=similarity_boost
    )
    
    # 保存音频片段
    filename = f"{output_dir / speaker}_{int(time.time())}.mp3"
    save(audio, filename)
    
    return filename

async def generate_full_podcast(segments: List[Dict[str, Any]], output_path: str):
    """生成完整的播客音频"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建临时目录存放音频片段
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # 生成所有音频片段
    audio_files = []
    for segment in segments:
        filename = await generate_audio_segment(segment, temp_dir)
        audio_files.append(filename)
    
    # 合并音频文件
    from pydub import AudioSegment
    
    combined = AudioSegment.empty()
    for file in audio_files:
        audio_segment = AudioSegment.from_mp3(file)
        combined += audio_segment
    
    # 保存最终文件
    combined.export(output_path, format="mp3")
    
    # 清理临时文件
    for file in audio_files:
        os.remove(file)
    os.rmdir(temp_dir)
    
    return output_path

async def create_podcast(podcast_theme: str, book_summary_path: str, duration_minutes: int, output_path: str):
    """创建完整的播客流程"""
    # 1. 读取书籍摘要
    book_summary = read_text_file(book_summary_path)
    
    # 2. 生成核心话题
    print("生成核心话题...")
    core_topics = generate_core_topics(podcast_theme, book_summary, duration_minutes)
    print(f"核心话题已生成:\n{core_topics}\n")
    
    # 3. 生成播客脚本
    print("生成播客脚本...")
    transcript = generate_podcast_transcript(book_summary, core_topics)
    print(f"播客脚本已生成:\n{transcript[:500]}...\n")
    
    # 4. 解析脚本
    segments = parse_transcript(transcript)
    print(f"解析出 {len(segments)} 个对话段落")
    
    # 5. 生成音频
    print("生成播客音频...")
    final_path = await generate_full_podcast(segments, output_path)
    print(f"播客已生成并保存至: {final_path}")
    
    return {
        "core_topics": core_topics,
        "transcript": transcript,
        "audio_path": final_path
    }

if __name__ == "__main__":
    # 示例用法
    book_summary_path = "path/to/book_summary.txt"
    duration_minutes = 15
    output_path = "output/podcast.mp3"
    podcast_theme = "How Social Media Ruined My Life (self-doubt):The negative impacts of social media on mental health and self-esteem, Body image issues"
    
    asyncio.run(create_podcast(podcast_theme, book_summary_path, duration_minutes, output_path))
