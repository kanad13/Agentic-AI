import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="Agentic-AI Chatbot", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.write("""
#### Agentic-AI Chatbot Demo: Tool Calling in Action

This is a chatbot that can answer your questions using information from documents, Wikipedia, and the internet. It's designed to be a helpful tool for finding information quickly.

**Click the chatbot page sidebar to access the chatbot.**

**How to Use the Chatbot:**

1.  **Ask a Question:** Type your question in the text box at the bottom of the chatbot window.
2.  **Press Enter or Click Send:** The chatbot will process your question and provide an answer.
3.  **Read the Answer:** The chatbot's response will appear in the chat window.

#### More details


Check out the code of the chatbot [on Github.](https://github.com/kanad13/LangChain-MultiTool-Agent)

Checkout my website for other AI/ML projects - [Kunal-Pathak.com](https://www.kunal-pathak.com).
				 """)
