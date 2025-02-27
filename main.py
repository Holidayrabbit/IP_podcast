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

import json


# 加载环境变量
load_dotenv()

# 设置 ElevenLabs API 密钥


# 设置 OpenRouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 定义模型
DEEPSEEK_MODEL = "google/gemini-2.0-pro-exp-02-05:free"
GEMINI_MODEL = "google/gemini-2.0-pro-exp-02-05:free"

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

def read_text_file(directory_path: str) -> str:
    """读取目录下所有文本文件的内容并合并"""
    combined_text = []
    
    # 确保路径存在
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"目录不存在: {directory_path}")
    
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    print(f"发现 {len(txt_files)} 个txt文件")
    
    # 按文件名排序
    txt_files.sort()
    
    # 读取每个文件
    for filename in txt_files:
        file_path = os.path.join(directory_path, filename)
        print(f"正在读取文件: {filename}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                combined_text.append(content)
        except Exception as e:
            print(f"读取文件 {filename} 时出错: {str(e)}")
    
    # 用换行符合并所有文本
    final_text = "\n\n".join(combined_text)
    print(f"合并后的文本总长度: {len(final_text)} 字符")
    
    return final_text

def generate_core_topics(podcast_theme: str, book_summary: str, duration_minutes: int) -> str:
    """使用DeepSeek生成核心话题"""
    prompt_template = """You are a professional podcast content strategist. Generate core topics for a two-person dialogue podcast based on the following inputs.

    Podcast Theme:
    {podcast_theme}

    Book Summary:
    {book_summary}

    Expected Duration: {duration_minutes} minutes

    Instructions:
    1. If duration is under 5 minutes, generate 3 topics. Otherwise, generate 5 topics.
    2. Topics should follow a logical progression from basics to practical applications to broader perspectives.
    3. Ensure topics are interconnected and flow naturally.
    4. Each topic must include a title and detailed explanation.

    Output format: Required JSON Structure:
    - Root object must contain "core_topics" array 
    - Each topic in core_topics must have:
      * core_topic (main topic title)
      * explanation (detailed description)

    Response must be valid JSON only, no additional text.
    Focus on creating engaging, discussion-worthy topics that match the theme and book content."""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | deepseek_model | StrOutputParser()
    
    result = chain.invoke({
        "book_summary": book_summary,
        "podcast_theme": podcast_theme,
        "duration_minutes": duration_minutes
    })
    
    # 打印API返回结果，用于调试
    print("API返回结果：")
    print(result)
    
    # 确保输出目录存在
    os.makedirs("output/core_topics", exist_ok=True)
    
    # 尝试解析JSON并保存
    try:
        # 解析JSON字符串
        topics_data = json.loads(result)
        # 保存到文件
        with open("output/core_topics/core_topics.json", "w", encoding="utf-8") as f:
            json.dump(topics_data, f, ensure_ascii=False, indent=4)
    except json.JSONDecodeError:
        # 如果不是有效的JSON，则直接保存文本
        with open("output/core_topics/core_topics.txt", "w", encoding="utf-8") as f:
            f.write(result)
    
    return result

def generate_podcast_transcript(book_summary: str, core_topics: str, ip_setting: str) -> str:
    """使用Gemini生成播客对话脚本"""
    prompt_template = """
    You are a professional podcast script writer. Based on the following book summary, core topics, character profiles, and conversation duration, please create a natural, engaging, and informative two-person dialogue podcast script.
    Requirements: Ensure the dialogue reflects each character's personality traits, and the conversation focuses on the book's content.

    Book Summary:
    {book_summary}

    Core Topics:
    {core_topics}

    Character Profiles:
    {ip_setting}

    Conversation Duration: {duration_minutes} minutes

    Please create the dialogue script in the following format (in actual dialogue, A and B should be replaced with their respective names):
    ****** opening ******
    A : (opening remarks)
    B : (response)
    ...

    ****** content ******
    A [emotion1]: (discussing first topic)
    B [emotion2]: (responding and deepening discussion)
    A [emotion3]: (raising questions or new perspectives)
    B [emotion4]: (responding)
    ...

    ****** content ******
    (continuing to next topic)
    ...

    ****** closing ******
    A : (summarizing discussion)
    B : (closing remarks)

    Requirements:
    1. The dialogue should flow naturally, like real people conversing
    2. Each speaker's tone and emotion should be noted in square brackets
    3. Include opening remarks, multiple content sections, and closing remarks
    4. Ensure coverage of all core topics
    5. Avoid lengthy monologues, maintain interactivity
    6. Total dialogue length and depth should be appropriate for a {duration_minutes}-minute podcast
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | gemini_model | StrOutputParser()
    
    # 生成对话脚本
    transcript = chain.invoke({
        "book_summary": book_summary,
        "core_topics": core_topics,
        "ip_setting": ip_setting,
        "duration_minutes": calculate_duration_from_topics(core_topics)
    })
    
    # 确保输出目录存在
    output_dir = os.path.join(".", "output", "transcript")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存对话脚本
    output_file = os.path.join(output_dir, "demo1_1.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    print(f"对话脚本已保存至: {output_file}")
    
    return transcript

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

async def create_podcast(podcast_theme: str, book_summary_path: str, ip_setting: str, duration_minutes: int, output_path: str):
    """创建完整的播客流程"""
    # 1. 读取书籍摘要
    book_summary = read_text_file(book_summary_path)
    
    # 2. 生成核心话题
    print("生成核心话题...")
    core_topics = generate_core_topics(podcast_theme, book_summary, duration_minutes)
    print(f"核心话题已生成:\n{core_topics}\n")
    
    # 3. 生成播客脚本
    print("生成播客脚本...")
    transcript = generate_podcast_transcript(book_summary, core_topics, ip_setting)
    print(f"播客脚本已生成:\n{transcript[:500]}...\n")
    
    # # 4. 解析脚本
    # segments = parse_transcript(transcript)
    # print(f"解析出 {len(segments)} 个对话段落")
    
    # # 5. 生成音频
    # print("生成播客音频...")
    # final_path = await generate_full_podcast(segments, output_path)
    # print(f"播客已生成并保存至: {final_path}")
    
    return {
        "core_topics": core_topics,
        "transcript": transcript,
        # "audio_path": final_path
    }

if __name__ == "__main__":
    # 示例用法
    book_summary_path = "./data/summary/self_improvement/"
    duration_minutes = 10
    output_path = "./output/podcast.mp3"
    podcast_theme = "How Social Media Ruined My Life (self-doubt):The negative impacts of social media on mental health and self-esteem, Body image issues"
    
    ip_setting = """
    Character Information:
        Character A:
            Image: Wise Elder
            Name: Samuel "Sam" Eldredge
            Character Profile:
            1. Keywords: Wisdom, Experience, Empathy, Humility, Humor, Slightly Rebellious "Old Soul"
            2. Background: Sam is a 74-year-old elder, but definitely not the traditional, preachy type. He's a "cross-domain sage" who has succeeded in multiple fields. In his youth, he was a rock musician, later became a psychology professor, and at 50, started a new business focusing on happiness psychology and personal growth. His interests are diverse: yoga, meditation, stand-up comedy... he even started learning programming at 70. His wisdom comes from his rich, bold, and sometimes rebellious life experiences, not just from books. He likes to teach through vivid stories, speaking casually but hitting the mark, always giving that "enlightening" feeling.
            - Tolerant and Authentic: Doesn't lecture audiences with clichés, preferring to communicate through humor and empathy, acknowledging his own failures and vulnerabilities.
            - Firm yet Inclusive: Has his own convictions about eternal "truths" (like how to live happily, how to view self-worth) but also accepts new trends and is willing to keep learning.
            - Playfully Rebellious: Deliberately "lightly mocks" modern personal development concepts, like "wake up at 5 AM to succeed" - he says having good tea and meditation at noon can lead to success too.
            - Far from Perfect: Sam admits his flaws, like focusing too much on material success in his youth and neglecting family; but he later mended these relationships, making him feel "authentically lived."
            4. Voice Characteristics: Warm baritone, magnetic, slow-paced but with firm power, as if every word flows from the depths of his soul with a touch of humor.
            5. Behavioral Traits:
            - Uses Poetry as Metaphors: When facing any life issues, he tends to use classical poetry or ancient texts as metaphors, which might amuse others, but he believes it makes the lessons more profound.
            - Collects Perpetual Calendars: Sam enjoys hunting for different types of perpetual calendars at various flea markets, which has become his passionate hobby.
            6. Flaws:
            - Sometimes Too Preachy: Despite good intentions, he occasionally slips into "teaching mode," making it feel like a lecture, especially when he thinks a young person needs guidance.
            - Stubborn Views: Sometimes he's stubborn about certain traditional views, refusing to accept new things, occasionally making Chloe laugh helplessly.
            7. Likes:
            - Nostalgic Black and White Films: Sam loves classic movies and regularly hosts small movie nights, inviting friends to watch together.
            - Tea and Coffee Culture: Has unique rituals for brewing tea at home, like specific teaware for different types of tea, particularly enjoying this process.
            8. Dislikes:
            - Fast-paced Lifestyle: Disapproves of modern people's fast-paced life, believing it makes people lose the joy of living.

        Character B:
            Image: Young Professional
            Name: Alex Morey
            Character Profile:
            1. Keywords: Passionate, Infectious Energy, Authentic, Diverse, Innovative, Occasionally Self-deprecating
            2. Background: Alex is a 27-year-old cross-domain professional who grew up in Los Angeles in a mixed-race family, with an entrepreneur father and a community psychologist mother. She studied Cognitive Science at Stanford University but chose not to follow a traditional career path, instead becoming a content creator focusing on personal growth and creative thinking. Alex is passionate about breaking free from "planned" constraints, focusing on "experimental growth" - she sees her life as a series of experiments, embracing both failures and successes. She likes to interpret complex topics about personal development and success in young people's language. Alex has both youthful sharpness and mature self-reflection. Despite her young age, she has already written a bestseller (AI-assisted, unknown to listeners) and been a speaker at several TEDx events.
            3. Personality Traits:
            - Authentic and Direct: Alex doesn't pretend to "know it all." When discussing topics like "finding meaning," she admits she's still exploring but shares her unique "trial and error" experiences.
            - Passionate and Infectious Activist: She constantly encourages listeners to take action, even the smallest step, believing "action itself defines direction."
            - Fearlessly Deconstructs Traditional Wisdom: She boldly challenges outdated personal development theories like "hard work equals success" or "your weaknesses will ruin you," supporting her views with modern cases or neuroscience research.
            - Light-hearted and Self-deprecating: She likes to add self-deprecating humor when sharing her experiences, like "I tried 8 morning routines, failed at 7, but the 8th somehow worked! Though I haven't fully stuck to it yet, hey, who can become perfect in a day?"
            - Slightly Rebellious but Vulnerable: She dislikes being "defined," sometimes questioning traditional life paths; but occasionally shares her own growth struggles openly.
            4. Voice Characteristics: Clear and pleasant young female voice, full of energy and approachability, slightly fast-paced but natural, sometimes raising pitch with excitement, engaging to listen to while capable of showing quiet and gentle sides.
            5. Special Behaviors:
            - Uses Colorful Post-its: Alex habitually writes positive reminders on post-its before new appointments or important events, sticking them on her computer or mirror for self-encouragement.
            - Names Her Plants: She has many potted plants, each with its own name (e.g., her favorite cactus is called "Pinpin"), and shares her emotions and daily life with them.
            6. Flaws:
            - Over-reliance on Technology: Alex tends to rely on phones and apps to handle life's problems, sometimes neglecting face-to-face real interactions.
            7. Likes:
            - Tech Innovation and Future Trends: Alex loves technology, curious about various new apps and tools, often trying and sharing her user experiences.
            - Fun Social Activities: She enjoys participating in various gatherings and social events, discovering new friends and interesting stories.
            8. Dislikes:
            - Fake Social Media Presentations: Dislikes those who display perfect lives on social media that aren't real, believing this behavior leads to unnecessary anxiety.

    Their Podcast Interaction Dynamic:
        Conversation Style: Sam and Alex's podcast interactions blend wisdom inheritance with youthful challenge. Sam often counters Alex's free-spirited views with his "old-school wisdom," while Alex challenges him with latest research or modern concepts.
        - Humor and Warmth: They often joke with each other. For example, Alex might tease Sam's "old-school" ways: "I can't do daily handwritten journals like Sam, my journal goes straight to the cloud!" While Sam might playfully critique Alex's life optimization: "Alex, how about enjoying life occasionally instead of trying to hack it forever?"
        - Mutual Appreciation: Despite their clashing viewpoints, they respect each other. Sam praises Alex's boldness and innovation, seeing infinite possibilities in the younger generation; while Alex admires Sam's life experience and gentle wisdom, finding comfort for her own growth.
    
    Their Meeting Story:
        Setting: A Coffee Shop in the Great Southwest
            After college graduation, Alex decided to take a Gap Year, searching for life direction through travel and conversations with strangers, seeking inspiration and answers about self-development. Meanwhile, Sam, retired for many years, was traveling internationally and writing his third book about "cross-generational wisdom transmission." They met in an unnamed town in the American Southwest, where a vintage, atmospheric coffee shop became their meeting point.
        Specific Encounter:
            Alex had just finished a long road trip, still uncertain about her future. She sat in the coffee shop with a laptop and notebook, doing her daily reflection and planning. At this time, Sam was in the corner, hand-writing his journal (old-school style, very eye-catching). As both appeared "immersed" in their writing activities, they caught each other's attention.
            Curious Alex, trying to break the lonely silence, noticed Sam's handwriting and mistaking him for just an eccentric artistic traveler, made a joke: "Who still writes journals by hand these days! Is this for some social media documentary?"
            Sam looked up and humorously replied:
            "When you reach my age, you'll find that social media can't remember every day you've lived."
            Alex was immediately struck by this response, sensing this elderly gentleman possessed wisdom beyond the ordinary, yet without a lecturing tone. She became interested in continuing the conversation with Sam, naturally leading to discussing her own search for life direction and meaning.

    """

    asyncio.run(create_podcast(podcast_theme, book_summary_path, ip_setting, duration_minutes, output_path))
