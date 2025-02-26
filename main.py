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
    确保话题之间有逻辑连贯性。

    输出的json格式如下：
    {
        "core_topics": [
            {
                "topic_title": "话题标题",
                "topic_description": "话题描述"
            }
        ]
    }
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

def generate_podcast_transcript(book_summary: str, core_topics: str, ip_setting: str, duration_minutes: int) -> str:
    """使用Gemini生成播客对话脚本"""
    prompt_template = """
    你是一位专业的播客脚本撰写者。请根据以下书籍摘要,核心话题,人设特点 对话时长，创建一段自然、有趣且信息丰富的双人对话播客脚本。
    要求：让对话的两个人的语言展现出他们的人格特点，另外对话的主题是即将输入的书籍的信息

    书籍摘要:
    {book_summary}

    核心话题:
    {core_topics}

    人设特点:
    {ip_setting}

    对话时长: {duration_minutes}分钟

    请按照以下格式创建对话脚本:
    ****** opening ******
    A : (开场白)
    B : (回应)
    ...

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
    A : (总结讨论)
    B : (结束语)

    要求:
    1. 对话应该自然流畅，像真实的人在交谈
    2. 每个发言者的语气和情感应在方括号中标注
    3. 包含开场白、多个内容部分和结束语
    4. 确保内容覆盖所有核心话题
    5. 避免过长的独白，保持互动性
    6. 总对话长度和深度应适合{duration_minutes}分钟的播客
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
    transcript = generate_podcast_transcript(book_summary, core_topics,ip_setting,duration_minutes)
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
    
    ip_setting = """
    两个人的个人信息如下：
        角色A:
            形象:智者老人 
            姓名:萨缪尔·"萨姆"·埃尔德里奇（Samuel “Sam” Eldredge）
            具体人设：
            1. 人设关键词：智慧、阅历、共情、谦逊、幽默感、略带叛逆的“老灵魂”
            2. 背景设定：萨姆是一位74岁高龄的老人，但绝对不是那种守旧的、带着说教腔调的角色。他是一位曾在多个领域取得过成功的“跨界智者”。年轻时，他曾当过摇滚乐手，后来成为一名心理学教授，又在50岁时重新创业，专注于研究幸福心理学和个人成长。他的兴趣广泛：瑜伽、冥想、单口喜剧……甚至在70岁时开始学习编程。他的智慧来自于他自己丰富、大胆且有时略显叛逆的人生经历，而不是单单依靠读书。他喜欢用生动的故事讲道理，语调轻松，却一针见血，总是有种“醍醐灌顶”的感觉。
            3. 个性特点
            - 宽容而真实：不会用“老生常谈”的方式劝导听众，更愿意通过幽默和共情与他们交流，承认自己的失败与脆弱。
            - 坚定又包容：对永恒的“真理”有自己的坚持（比如如何活得快乐、如何看待自我价值），但也接受新思潮，愿意不断学习。
            - 略带叛逆的幽默：会故意“轻吐槽”一些现代流行的个人发展概念，比如“5点起床才能成功”——他说自己认为中午喝杯好茶和冥想也能让人成功。
            - 绝非完美圣人：萨姆承认他有缺点，比如年轻时太过追求物质成功，忽略了家人；但后来弥补了这些关系，这样的坦诚让人觉得他“活得真实”。
            4. 声音表现：温暖的男中音，富有磁性，语调缓慢但带有坚定的力量，仿佛每句话都是从他的灵魂深处流淌出来的笑谈。
            5. 行为特点：
            - 习惯用诗句反打比方：他遇到任何生活中的问题时，总会用古诗或古文的语言来进行比喻，虽然只会让旁人感到好笑，但他觉得这样会让道理更深刻。
            - 爱收集万年历：萨姆在各类杂货市场上喜欢寻找不同种类的万年历，这成为他一段沉迷的爱好。
            6. 缺点：
            - 有时说教过头：尽管善意，他有时会在不经意间进入“教导模式”，让人感觉听课的味道，尤其是当他觉得某个年轻人需要引导时。
            - 固执己见：他有时在某些传统观念上显得固执，拒绝接受新事物，偶尔会让克洛伊哭笑不得。
            7. 喜欢的东西：
            - 怀旧的黑白电影：萨姆热爱经典电影，并定期会在週末举办小型电影之夜，邀请朋友们一起观看。
            - 茶道和咖啡文化：对于在家泡茶时的仪式感，萨姆有独特的仪式，比如每种茶叶预备的不同茶具，他特别享受这个过程。
            8. 讨厌的东西：
            - 快节奏的生活方式：他对现代人的快节奏生活有些不满，认为这让人们失去了享受生活的乐趣。

        角色B:
            形象：年轻人
            姓名：艾利克斯·莫雷（Alex Morey）
            具体人设：
            1. 人设关键词：热情、有感染力、真实、多元化、创新、偶尔自嘲
            2. 背景设定：艾利克斯是一个27岁的跨领域年轻人，成长于洛杉矶一个混血家庭，父亲是企业家，母亲是一位社区心理医生。大学时，她在斯坦福大学学习了认知科学，但没选择传统的职业道路，而是成为了一名个人成长和创意思维方面的内容创作者。艾利克斯热衷于探索如何打破“计划”的束缚，专注于“实验式成长”——她将自己的人生看作一场接一场大大小小的实验，失败和成功都尽可能被拥抱。她喜欢用年轻人的语言来解读关于个人发展和成功的复杂话题。艾利克斯既有少年人的锐气，又充满自省的成熟。尽管她年纪轻轻，但她居然已经写了一本畅销书（由ChatGPT辅助创作，听众并不知道），并成为数次TEDx演讲的主讲人。
            3. 个性特点
            - 真实而直率：艾利克斯不假装“全部都了解”。比如在谈论如何“找到意义”这类话题时，她会承认自己也仍在探索，但会分享她独特的“尝试和犯错”经验。
            - 热情和有感染力的行动主义者：她会不断鼓励听众付诸行动，哪怕是最小的一步，因为她相信“行动本身会定义方向”。
            - 无畏地拆解传统智慧：她大胆挑战一些过时的个人发展理论，比如“努力工作就能成功”“你的弱点会毁掉你”，并用现代案例或脑科学研究支撑她的观点。
            - 轻松有趣，偶尔自嘲：她喜欢在讲述自己的经历时加入很多自嘲，比如“我试过了8种晨间日程，其中七种完全搞砸了，但第八种居然有用！虽然还没完全坚持下来，但嘿，谁能一天就变成完美的人呢？”
            - 略显叛逆但带着脆弱：她不喜欢“被定义”，有时会质疑传统的生活轨迹；但偶尔也会坦诚自己在成长过程中遇到的迷茫情绪。
            4. 声音表现：清晰悦耳的女青年嗓音，充满活力与亲和力，节奏稍快但很自然，有时会因为激情略微提高音调，让人听得十分带劲，但同时也能展现出安静和温柔的一面。
            5. 特色行为：
            - 用彩色便利贴记录生活：Chloe习惯在每个新约会或重要事情前写下积极的提示，贴在她的电脑上或镜子上，以鼓励自己。
            - 喜欢给植物起名字：她有很多盆栽植物，每一棵都有自己的名字（例如，喜欢的仙人掌叫“钉钉”），并和它们分享自己的情感和日常。
            6. 缺点：
            - 过于依赖科技：Chloe容易依赖手机和应用程序来处理生活中的问题，有时忽略了面对面的真实互动。
            7. 喜欢的东西：
            - 科技创新和未来趋势：Chloe热爱科技，对各种新应用和工具充满好奇，常常尝试并分享她的使用体验。
            - 有趣的社交活动：她喜欢参加各种聚会和社交活动，发掘新朋友和好玩的故事。
            8. 讨厌的东西：
            - 虚伪的社交媒体表现：对那些在社交媒体上展示完美生活但实际上并不真实的人感到厌烦，认为这种行为会导致人们无谓的焦虑。

    两个人在播客中的关系设定与互动如下：
        对话风格：萨姆和艾利克斯在播客里的互动，既有智慧的传承，也有年轻人的挑战。萨姆常常用自己的“老派智慧”反驳克洛伊奔放的观点，而艾利克斯则用最新的研究或新潮的理念跟他“争论”。
        - 幽默与亲切：两人偶尔会开彼此的玩笑。比如，艾利克斯可能会调侃萨姆的“老派”方式：“我没办法像萨姆那样每天手写日记，我的日记直接上传到了云端！” 而萨姆可能会轻吐槽艾利克斯总是在生活中“过度优化”：“艾利克斯，偶尔享受生活，而不是试图永远黑客它，如何？”
        - 彼此欣赏：尽管有观点碰撞，但他们彼此尊重。萨姆会称赞艾利克斯的大胆和创新，认为年轻一代有无限可能；而艾利克斯则敬佩萨姆的人生阅历和温柔的智慧，为自己的成长找到安慰。
    
    两个人的互动故事如下：
        背景环境：大西南的咖啡馆
            艾利克斯在大学毕业后决定度过一段“间隔年”（Gap Year），她一直在寻找人生方向，计划通过旅行与陌生人交流，寻找更多关于自我发展的灵感和答案。与此同时，萨姆已经退休多年，正在跨国旅行并撰写他的第三本书，关于“跨世代的智慧传递”。两人相遇在美国西南部一个不知名的小镇，在那里，一家复古而富有意境的咖啡馆成为他们的交汇点。
        具体相遇经过：
            艾利克斯刚刚结束了一次长途公路旅行，对未来依然迷茫。她坐在咖啡馆里，带着一台笔记本电脑和本子，在做每日的反思笔记和计划。此时，萨姆正在咖啡馆的角落低头手写日记（老派风格，非常扎眼）。由于两人都显得“沉浸”于自己的读写活动，彼此吸引了目光。
            好奇的艾利克斯，试图打破孤独的沉默，看到萨姆的手写内容，误以为这个老太爷只是个古怪的文艺旅行者，于是开了一个玩笑：“这年头还有人用笔写日记呢！这些是不是给某个社交媒体纪录片用的啊？”
            萨姆抬起头，幽默地回了一句：
            “等你到了我这个年纪，你会发现，社交媒体根本记不住你活过的每一天。”
            艾利克斯立刻被这句话“击中”，感受到这个老先生似乎拥有超越日常的“智慧”，但又不带说教的语气。她感兴趣地继续和萨姆搭话，自然而然谈到她自己在寻找人生方向和意义。

    """

    asyncio.run(create_podcast(podcast_theme, book_summary_path, ip_setting,duration_minutes, output_path))
