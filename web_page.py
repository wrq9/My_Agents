from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

os.environ["DEEPSEEK_API_KEY"] = 'sk-d69980bdd41a4d228e3f783d4d92a246'

@tool
def web_loader(url: str) -> str:
    """抓取url对应网页的内容"""
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs[0].page_content

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

tools = [web_loader]
llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "请阅读给定的网页内容，并用中文向我介绍GRPO的原理。"),
    ("placeholder", "{chat_history}"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm_with_tools, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
url = "https://huggingface.co/docs/trl/main/grpo_trainer"
result = agent_executor.invoke({"input": url})

print(result)