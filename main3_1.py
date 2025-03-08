import os
import json
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

# 创建 LangChain 模型实例
model = ChatOpenAI(
    model=QWEN2_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

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

def extract_core_topics(text_content: str) -> str:
    """使用AI提取文本的核心话题，让AI自行决定话题数量"""
    prompt_template = """You are a professional content analyst. Please extract the MOST IMPORTANT core insights from the text and generate key topics that will be used to create podcast dialogue scripts later.

    Text Content:
    {text_content}

    Requirements:
    1. Extract only the MAJOR core topics from the text content
    2. Focus on high-level, broad themes rather than specific details
    3. Combine related ideas into single comprehensive topics
    4. Be selective - only include truly essential topics (typically 3-7 topics total)
    5. The number of topics should be appropriate for the content - shorter or simpler books may need fewer topics
    6. Each topic should include a concise title and detailed explanation
    7. Topics should have logical connections and flow between them

    Output Format Requirements: Must be valid JSON with the following structure.
    Note: The structure below uses [ ] for illustration, actual output should be proper JSON:

    [
        "core_topics": [
            [
                "topic": "Topic Title",
                "explanation": "Detailed Explanation",
                "transition": "Transition to the next topic",
                "logical_structure": "Logical structure of this topic in the podcast"
            ],
            ...additional topics as appropriate...
        ]
    ]

    Return only the JSON content, no additional text."""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model | StrOutputParser()
    
    result = chain.invoke({
        "text_content": text_content
    })
    
    return result

def save_topics_to_json(topics_data: str, output_path: str):
    """保存话题到JSON文件"""
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 解析JSON字符串
        topics = json.loads(topics_data)
        
        # 保存到文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(topics, f, ensure_ascii=False, indent=4)
        
        print(f"核心话题已保存至: {output_path}")
        return True
    except json.JSONDecodeError:
        print("JSON解析错误，保存为文本文件")
        with open(output_path.replace('.json', '.txt'), "w", encoding="utf-8") as f:
            f.write(topics_data)
        return False
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")
        return False

def process_book_summary(input_file: str, output_file: str):
    """处理书籍摘要并提取核心话题"""
    # 读取文本内容
    content = read_text_file(input_file)
    if not content:
        print("无法读取文件内容")
        return
    
    # 提取核心话题
    print("正在提取核心话题...")
    topics = extract_core_topics(content)
    
    # 保存结果
    success = save_topics_to_json(topics, output_file)
    if success:
        print("核心话题提取完成！")
    else:
        print("核心话题提取完成，但保存时出现问题")

if __name__ == "__main__":
    # 示例用法
    input_file = "./data/summary/self_improvement/how-to-break-up-with-your-phone.txt"  # 输入文件路径
    output_file = "./output/core_topics/book1_topics.json"    # 输出文件路径
    
    process_book_summary(input_file, output_file)
