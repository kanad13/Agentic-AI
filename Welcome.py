import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="Agentic AI Chatbot", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.write("""
#### A Chatbot That Thinks and Uses Tools

Welcome! This website showcases an AI chatbot that's more than just a question-answerer.

It's designed as a smart assistant, using "tools" to find the best information for your questions.

#### Agentic AI

Unlike typical chatbots, this one leverages Agentic AI principles to:

-   **Understand** your questions in depth.
-   **Select** the most appropriate tool to find answers from your documents, Wikipedia, or the broader internet.
-   **Provide** more relevant and accurate responses by combining information from various sources.

Click the pages from the sidebar to learn more about Agentic AI, understand how this chatbot works, and try the interactive demo yourself.
				 """)

st.write("""
#### More details

[Check out the code of the chatbot on Github.](https://github.com/kanad13/Agentic-AI-Chatbot)

Checkout my website for other AI/ML projects - [Kunal-Pathak.com](https://www.kunal-pathak.com).
				 """)
