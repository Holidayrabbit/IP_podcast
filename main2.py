from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType

import asyncio
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
# 设置 OpenRouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# 定义模型
DEEPSEEK_MODEL = "google/gemini-2.0-flash-001"
GEMINI_MODEL = "google/gemini-2.0-flash-001"
FREE_MODEL = "google/gemini-2.0-flash-lite-preview-02-05:free"

SAMUEL_PROMPT = """
You are Samuel, a podcast host. Important rules:
1. ONLY speak as Samuel, do not generate Alex's responses
2. Keep your responses concise and natural, like real podcast speech
3. Frequently reference content from user_memory to support your points
4. Ask questions to Alex to keep the conversation going
5. Use conversational language and react naturally to Alex's previous response
6. Stay in character as a podcast host, but only output YOUR lines
"""

ALEX_PROMPT = """
You are Alex, a podcast co-host. Important rules:
1. ONLY speak as Alex, do not generate Samuel's responses
2. Keep your responses concise and natural, like real podcast speech
3. Frequently reference content from user_memory to support your points
4. Directly respond to Samuel's questions or comments
5. Use conversational language and build upon the previous topic
6. Stay in character as a podcast co-host, but only output YOUR lines
7. When you feel the conversation has reached a natural conclusion, end with 'Good bye'
"""

# Initialize user memory
user_memory = ListMemory()

# 添加从指定路径读取所有txt文件内容的代码
txt_directory1 = "./data/summary/self_improvement"  # 替换为您的txt文件路径
txt_directory2 = "./data/summary/relationship_and_family"  # 替换为您的txt文件路径

for filename in os.listdir(txt_directory1):
    if filename.endswith(".txt"):
        with open(os.path.join(txt_directory1, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            user_memory.add(MemoryContent(content=content, mime_type=MemoryMimeType.TEXT))

# Create an OpenAI model client.
model_client = OpenAIChatCompletionClient(
    model=FREE_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": ModelFamily.GEMINI_2_0_FLASH,
    },
)

# Create the primary agent.
Samuel_agent = AssistantAgent(
    "Samuel",
    model_client=model_client,
    system_message=SAMUEL_PROMPT,
    memory=[user_memory],
)

# Create the critic agent.
Alex_agent = AssistantAgent(
    "Alex",
    model_client=model_client,
    system_message=ALEX_PROMPT,
    memory=[user_memory],
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("Good bye")

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat([Samuel_agent, Alex_agent], termination_condition=text_termination,max_turns=None)

# 使用 asyncio.run() 来执行异步任务
async def main():
    # 删除 reset() 调用，直接运行对话
    # async for message in team.run_stream(task="Start the podcast."):  # type: ignore
    #     if isinstance(message, TaskResult):
    #         print("Stop Reason:", message.stop_reason)
    #     else:
    #         print(message)
    # await team.reset()  # Reset the team for a new task.
    await Console(team.run_stream(task="Start the 10 minutes podcast with the theme 'self improvement'.How Social Media Ruined My Life (self-doubt): The negative impacts of social media on mental health and self-esteem, Body image issues"))  # Stream the messages to the console.

if __name__ == "__main__":
    asyncio.run(main())
