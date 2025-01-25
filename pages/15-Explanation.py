import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="Agentic AI and Tool Calling", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.write("""
#### On this page

This page explains how the chatbot uses Agentic AI & Tool Calling for making intelligent decisions.

#### How does the Agentic chatbot work?

- **It Chooses Tools:** When you ask a question, the chatbot decides if it needs to use special tools to find the answer. It doesn't just guess!
- **It Acts on its Own:** It can automatically use these tools:
  - **Document Tool:** To read stories or documents you give it.
  - **Wikipedia Tool:** To check Wikipedia for facts.
  - **Internet Tool:** To search the web for up-to-date info.
- **It Has a Goal:** The chatbot tries its best to answer your questions using all the resources it has.
				 """)

st.image('assets/tool-selection.png', caption=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.write("""
#### Taking Actions

To show this "action" capability, the chatbot does something a real assistant might do: it can create a report on GitHub.

You can see some of the questions and answers that the chatbot has answered here - `link to github repo issues tab`
				 """)
