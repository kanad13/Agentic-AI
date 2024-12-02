import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="LangChain-MultiTool-Agent", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.write("""
#### LangChain-MultiTool-Agent

An intelligent chatbot built with LangChain that dynamically selects the appropriate tool to answer user queries.

It leverages the power of LangChain agents, retrieval-augmented generation (RAG), web search, and Wikipedia queries for context-aware, real-time responses.

#### More details


Check out the code of the chatbot [on Github.](https://github.com/kanad13/LangChain-MultiTool-Agent)

Checkout my website for other AI/ML projects - [Kunal-Pathak.com](https://www.kunal-pathak.com).
				 """)
