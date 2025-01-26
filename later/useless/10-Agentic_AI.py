import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="Agentic AI and Tool Calling", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.write("""
#### Understanding Smart AI Assistants

Imagine an AI that behaves like an intelligent assistant.

Agentic AI is a step towards creating  assistants - AI systems that can think, plan, and act to help you with complex tasks.
				 """)

st.image('assets/movie-night.png', caption=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.write("""
#### What is Agentic AI?

Agentic AI focuses on making AI systems more proactive and intelligent.

Agentic AI can:

- **Understand Information:** Analyze information from its surroundings, like text, web pages, and documents.
- **Plan Actions:**  Develop plans to achieve a goal, such as answering your question or completing a specific task.
- **Act Using Tools:**  Take action autonomously by using "tools" to gather information or perform operations.
- **Learn and Improve:**  Adapt and enhance its performance over time through experience.
				 """)

st.image('assets/agentic-decision.png', caption=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.write("""
#### What is Tool Calling?

To be genuinely helpful, a smart assistant needs access to the right resources. "Tool Calling" is the mechanism that provides an AI chatbot with access to these resources, which we call "tools."

"Tool Calling" helps the AI to:

1. **Recognize the Need:** Understand when it requires external information to answer your question.
2. **Choose the Best Tool:** Select the most appropriate tool for the task at hand (e.g., internet search for current events).
3. **Utilize the Tool:**  Effectively use the chosen tool to retrieve the necessary information.
4. **Provide Enhanced Answers:**  Integrate the information from the tool to give you more accurate, comprehensive, and relevant answers.
				 """)

st.image('assets/simple-agentic.png', caption=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
