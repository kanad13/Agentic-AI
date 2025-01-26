import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="Agentic AI Chatbot", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.write("""
#### Welcome to Agentic AI: AI that Acts

You are familiar with AI assistants like Siri or Alexa that respond to your questions.  Agentic AI is a step beyond this.

**Agentic AI is designed to take actions, not just react.**

Consider this difference:

- **Typical AI (like Siri):** You ask "What is the weather?" and it replies. It waits for your questions.
- **Agentic AI:** You might say "Plan a beach trip next weekend." Agentic AI would then perform actions for you.  It could:
    - **Research** beaches.
    - **Check** flight and hotel availability.
    - **Compare prices.**
    - **Book reservations (with your approval).**

**The core idea is that AI can set goals and act to achieve them independently.**

---

#### The Agentic Chatbot I built

You are right now viewing an Agentic AI chatbot that I built to demonstrate basic Agentic AI concepts.

Navigate to the "Agentic Chatbot" page in the sidebar.  Ask questions to see how it finds information and demonstrates agentic behavior.

---

#### How My Chatbot Demonstrates Agentic Behavior

My chatbot acts as a basic agent by:

- **Deciding how to answer:**
    - When you ask a question, the chatbot determines the best way to find the answer.
- **Accessing Information:** It can automatically:
    - Read provided documents.
    - Check Wikipedia.
    - Search the internet.
- **Agentic Action:**
    - The chatbot does not just stop at finding information, but it also performs actions in your behalf.
    - The [chatbot creates issues in Github](https://github.com/kanad1323/agentic-ai-output/issues) based on the interactions it had with you and the instructions you gave.

				 """)
