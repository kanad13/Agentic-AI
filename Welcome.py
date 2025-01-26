import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="Agentic AI Chatbot", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.write("""
#### Welcome to Agentic AI: AI That Acts

You are likely familiar with AI like Siri or Alexa that responds to your questions.  Agentic AI is a step beyond this.

**Agentic AI is designed to take actions, not just react.**

Consider this difference:

- **Typical AI (like Siri):** You ask "What is the weather?" and it replies. It waits for your instruction.
- **Agentic AI:** You might say "Plan a beach trip next weekend." Agentic AI would then perform actions for you.  It could:
    - **Research** beaches.
    - **Check** flight and hotel availability.
    - **Compare prices.**
    - **Book reservations (with your approval).**

**The core idea is AI that can set goals and act to achieve them independently.**

---

#### Examples of Agentic AI Concepts

You might have heard of Agentic AI in examples like:

- **Claude's "Computer Use":**  This lets you ask Claude to perform tasks directly on a computer, like ordering food online. [Youtube Video](https://youtu.be/ODaHJzOyVCQ?t=37)
- **OpenAI's "Operator":** This AI can browse the internet to complete tasks for you, such as booking flights. [Youtube Video](https://youtu.be/CSE77wAdDLg?t=136)

These are examples of AI designed to take actions to assist you.

---

####  This Chatbot Demonstrates Agentic AI Basics

This chatbot I built is a simplified example to show these basic concepts.  It demonstrates how AI can be designed to take actions in a clear way.

It is a practical demonstration, not a fully advanced agent, designed for understanding the fundamental principles.

---

####  How This Chatbot Works

This chatbot acts like a basic agent by doing these things:

* **Decision Making:**  When you ask a question, the chatbot decides how to find an answer.
* **Information Access:** It can automatically:
    * **Read documents** that you provide.
    * **Check Wikipedia** for information.
    * **Search the internet** for current information.
* **Goal Orientation:** The chatbot aims to answer your questions using these methods.

---

####  Agentic Action Example

To show action, this chatbot can create a report on GitHub.  This illustrates AI performing tasks, not just providing information.

See examples of questions and chatbot responses here: [Chatbot Question Examples](https://github.com/kanad1323/agentic-ai-output/issues).

---

####  Explore Agentic AI

Navigate to the "Agentic Chatbot" page in the sidebar.  Ask questions to see how it finds information and demonstrates agentic behavior.

				 """)

st.write("""
#### Learn More and Build Your Own

[Check out the chatbot code on Github.](https://github.com/kanad13/Agentic-AI)

Checkout my website for other AI/ML projects - [Kunal-Pathak.com](https://www.kunal-pathak.com).
				 """)
