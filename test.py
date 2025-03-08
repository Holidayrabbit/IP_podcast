from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Initialize user memory
user_memory = ListMemory()

# Add user preferences to memory
await user_memory.add(MemoryContent(content="The weather should be in metric units", mime_type=MemoryMimeType.TEXT))

await user_memory.add(MemoryContent(content="Meal recipe must be vegan", mime_type=MemoryMimeType.TEXT))


async def get_weather(city: str, units: str = "imperial") -> str:
    if units == "imperial":
        return f"The weather in {city} is 73 °F and Sunny."
    elif units == "metric":
        return f"The weather in {city} is 23 °C and Sunny."
    else:
        return f"Sorry, I don't know the weather in {city}."


assistant_agent = AssistantAgent(
    name="assistant_agent",
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o-2024-08-06",
    ),
    tools=[get_weather],
    memory=[user_memory],
)