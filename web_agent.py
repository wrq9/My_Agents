import streamlit as st
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

@st.cache_resource
def get_llm():
    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

llm = get_llm()

tools = [web_loader]
llm_with_tools = llm.bind_tools(tools)
need = st.text_area("需要我做什么？🧐")

prompt = ChatPromptTemplate.from_messages([
    ("system", need),
    ("placeholder", "{chat_history}"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm_with_tools, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
url = st.text_area("需要我浏览哪个网页？👨🏻‍💻")
if st.button("回复"):
    result = agent_executor.invoke({"input": url})
    output = result['output']
    st.write(output)