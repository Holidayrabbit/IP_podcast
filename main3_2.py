import os
import json
import random
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

# 设置 OpenRouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 定义模型
GEMINI_MODEL = "google/gemini-2.0-flash-001"
QWEN2_MODEL = "qwen/qwq-32b:free"
MYTHO_MODEL = "gryphe/mythomax-l2-13b"
CLAUDE_MODEL = "anthropic/claude-3.7-sonnet"

# 创建 LangChain 模型实例
model = ChatOpenAI(
    model=CLAUDE_MODEL,
    temperature=0.8,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# 角色设定
ip_setting1 = """
(important)Character Information:
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
"""

def read_text_file(file_path: str) -> str:
    """读取单个文本文件的内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            print(f"成功读取文件: {file_path}")
            print(f"文本长度: {len(content)} 字符")
            return content
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return ""

def generate_podcast_script(book_summary: str, core_topics: str, selected_topic: str, ip_setting: str, duration_minutes: int = 5) -> str:
    """生成播客脚本"""
    prompt_template = """
    You are a professional podcast script writer. 
    please create a part of a two-person dialogue podcast script based on the following book summary, core topics, selected topic, character profiles, and conversation length limit.

    Book Summary:
    {book_summary}

    Core Topics:
    {core_topics}

    Selected Topic:
    {selected_topic}

    Character Profiles:
    {ip_setting}

    Conversation length limit: approximately {word_count} words

    Please create the dialogue script in the following format:

    ****** content of {selected_topic} ******
    Samuel: ...
    Alex: ...
    Samuel: ...
    Alex: ...
    ...

    Requirements:
    1. (Attention)You only need to generate the content of the selected topic, don't include the opening and closing words of the podcast, and don't talk about other core topics.
    2. (Attention)You need to generate the content of the selected topic, following the logic of :
        - Propose new topics and their definitions: It is important to indicate that the previous topic has ended and this is a new topic.;
        - main contents of the selected topic：mainly based on the book summary and the two hosts'story;
        - conclusion of the selected topic: summarize the main points of the selected topic;
        - transition to the next core topic: find a natural transition to the next core topic.
    3. (Important)Ensure the characters stay true to their personalities and the conversation style. They can add their experiences and opinions to the content.
    4. (Important)Avoid lengthy monologues, maintain interactivity
    5. Use simple txt format without any markdown formatting

    Here is some examples of dialogue techniques :
    1. Propose new topics :
        -OK,let's start our first topic: ...
        -Now, let’s turn our attention to today's second content: ...
    2. Main content:    
        **Narrative Logic：**
            <talk experience/book content/story>: I will tell you, as you know, I grew up in a challenging background. ...
            <Extract more general questions and point out the point of view>: And what I mean by that is so many people make judgments about people. And they don't recognize the paths or the problems or the adversities they face....
            <after a deep conversation, go back to the mainline>：I want to go back to the moment in the magic shop that truly shaped you forever and change the trajectory of your life and so the person who's listening can really perhaps see themselves or someone they love in the story. 
        **Simplify concept：**
            <Explain to the audience in simpler, more direct words>:If you will. I want to see if I can translate back. This is where a lot of big words, I realize that you are a neurosurgeon and you teach these concepts all over the world, but I want you and I to put our arm around the person that's listening, and I want to see if I can't shorthand all this science, 
            <interrupt the other person briefly to ask questions>：
            A: And as a scientific serious researcher, I know that there is something called the bio-psycho-social models. 
            B: Okay, hold on. What is the bio, social, pycho? what did you say?
            Guest: bio-psycho-social model
            A: okay, and what does that mean in a normal person speak? 
            B: bio-psycho-social is.......
    3. Transition to the next core topic:
        -In summary, the core of [Topic 1] is… But the story is not over yet, because next we have to see how it affects [Topic 2].
        -If [Topic 1] is a glass of strong liquor, then [Topic 2] is a cup of refreshing tea that leaves you with a sweet aftertaste...
        -If you are interested in [Topic 1], then the following [Topic 2] will give you even more surprises!
    """
    
    # 估算字数（每分钟约150字）
    word_count = duration_minutes * 130
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model | StrOutputParser()
    
    # 生成对话脚本
    script = chain.invoke({
        "book_summary": book_summary,
        "core_topics": core_topics ,   
        "selected_topic": selected_topic,
        "ip_setting": ip_setting,
        "duration_minutes": duration_minutes,
        "word_count": word_count
    })
    
    return script

def save_script_to_file(script: str, output_path: str):
    """保存脚本到文件"""
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存到文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(script)
        
        print(f"播客脚本已保存至: {output_path}")
        return True
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")
        return False

def generate_podcast_from_topic(book_summary_path: str, core_topic_path: str, output_path: str, duration_minutes: int = 5):
    """根据书籍摘要和核心话题生成播客脚本"""
    # 读取书籍摘要
    book_summary = read_text_file(book_summary_path)
    if not book_summary:
        print("无法读取书籍摘要")
        return
    
    # 读取核心话题
    core_topic = read_text_file(core_topic_path)
    if not core_topic:
        print("无法读取核心话题")
        return
    
    # "The Addictive Design of Smartphones and Apps"
    # "Cognitive and Mental Health Impacts of Excessive Phone Use"
    # "Breaking Up with Your Phone: A Practical Approach"
    # "From FOMO to JOMO: Reclaiming Real-Life Connections"
    # "Advocacy for Mindful Technology Use"
    # "Building Sustainable Long-Term Habits"


    selected_topic = """
    "Building Sustainable Long-Term Habits"
    """

    # 生成播客脚本
    print("正在生成播客脚本...")
    script = generate_podcast_script(book_summary, core_topic, selected_topic, ip_setting1, duration_minutes)
    
    # 保存脚本
    success = save_script_to_file(script, output_path)
    if success:
        print("播客脚本生成完成！")
    else:
        print("播客脚本生成完成，但保存时出现问题")
    
    return script

if __name__ == "__main__":
    # 示例用法
    book_summary_path = "./data/summary/self_improvement/how-to-break-up-with-your-phone.txt"  # 书籍摘要路径
    output_path = "./output/podcast_scripts39/phone_addiction_topic_podcast6.txt"  # 输出脚本路径
    core_topics_path = "./output/core_topics/book1_topics.txt"
    duration_minutes = 10 # 播客时长（分钟）
    
    generate_podcast_from_topic(book_summary_path, core_topics_path, output_path, duration_minutes)
