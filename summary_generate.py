import os
from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import glob
import time

# 加载环境变量
load_dotenv()

# 设置 OpenRouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_MODEL = "google/gemini-flash-1.5"

CONTEXT_LENGTH = 900000
OUTPUT_LENGTH = 5000

# 创建 LangChain 模型实例
gemini_model = ChatOpenAI(
    model=GEMINI_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

def extract_text_from_pdf(pdf_path: str) -> str:
    """从PDF文件中提取文本"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = 15000) -> List[str]:
    """将文本分割成较小的块"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_chunk(chunk: str, target_length: int = OUTPUT_LENGTH, max_retries: int = 3) -> str:
    """使用Gemini模型总结文本块"""
    prompt_template = """
    请对以下文本内容进行总结。保持关键信息和核心观点，同时确保总结的内容清晰、连贯。
    如果遇到了了新的章节，请在结构上换行并添加新的章节标题。最终总结的长度应该接近{target_length}个字符。
    文本内容:
    {text}

    要求：总结的每一条都要有标题，而且内容要翔实，不要过于简略。
    总结应该保持原文的专业性和准确性。
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | gemini_model | StrOutputParser()
    
    for attempt in range(max_retries):
        try:
            result = chain.invoke({
                "text": chunk,
                "target_length": target_length
            })
            print(f"API 返回结果：\n{result}\n")
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"尝试 {attempt + 1} 失败，等待 5 秒后重试...")
            print(f"错误信息: {str(e)}")
            time.sleep(5)

def combine_summaries(summaries: List[str], target_length: int = 30000) -> str:
    """合并并优化多个总结，确保最终长度接近目标长度"""
    prompt_template = """
    请将以下多个文本总结合并为一个连贯的总结。最终总结的长度应该接近{target_length}个字符。

    文本总结:
    {summaries}

    要求：
    1. 保持内容的连贯性和逻辑性
    2. 删除重复的信息
    3. 确保关键观点都被保留
    4. 适当调整内容长度，使最终总结接近{target_length}个字符
    5. 保持专业性和准确性
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | gemini_model | StrOutputParser()
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = chain.invoke({
                "summaries": "\n\n".join(summaries),
                "target_length": target_length
            })
            if result is None:
                raise ValueError("API 返回为空")
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"合并总结时失败（尝试 {attempt + 1}/{max_retries}），等待 10 秒后重试...")
            print(f"错误信息: {str(e)}")
            time.sleep(10)  # 增加等待时间，避免频繁请求

def generate_book_summary(pdf_paths: List[str], output_path: str, target_length: int = OUTPUT_LENGTH ):
    """生成多本书的综合摘要"""
    all_summaries = []
    
    for pdf_path in pdf_paths:
        print(f"处理文件: {pdf_path}")
        
        # 提取文本
        text = extract_text_from_pdf(pdf_path)
        
        # 分块处理
        chunks = chunk_text(text)
        
        # 总结每个块
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"处理第 {i+1}/{len(chunks)} 个文本块...")
            summary = summarize_chunk(chunk)
            chunk_summaries.append(summary)
        
        # 合并该书的所有块总结
        book_summary = combine_summaries(chunk_summaries, target_length // len(pdf_paths))
        all_summaries.append(book_summary)
    
    # 合并所有书的总结
    final_summary = combine_summaries(all_summaries, target_length)
    
    # 保存最终总结
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_summary)
    
    print(f"摘要已生成并保存至: {output_path}")
    return final_summary

if __name__ == "__main__":
    # 示例用法
    pdf_paths = glob.glob("./books/test/*.pdf")
    
    output_path = "output/book_summary/self_improvement/book_summary.txt"
    
    summary = generate_book_summary(pdf_paths, output_path)
