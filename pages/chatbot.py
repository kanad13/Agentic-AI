import streamlit as st
from dotenv import load_dotenv
import os
import uuid
import bs4
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Initialize necessary components
config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="ChatBot", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')

# Define the model
model = ChatOpenAI(model="gpt-4o-mini")

# Additional configurations
does_model_support_streaming = True
os.environ["TOKENIZERS_PARALLELISM"] = "true"
embedding_batch_size = 512

# Define RAG tool
@st.cache_resource
def load_documents():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))},
    )
    return loader.load()

docs = load_documents()

# Split webpage data
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Compute embeddings and index them
@st.cache_resource
def create_vectorstore():
    embedding_wrapper = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={'batch_size': embedding_batch_size}
    )
    return FAISS.from_documents(documents=all_splits, embedding=embedding_wrapper)

vectorstore = create_vectorstore()

# Create a retriever object
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Define tools
retriever_tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)

search = TavilySearchResults(max_results=2)

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search, wikipedia, retriever_tool]

# Setup memory
memory = MemorySaver()

# Use threads
unique_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": unique_id}}

# Create agents
agent_executor_with_memory = create_react_agent(model, tools, checkpointer=memory)

# Custom prompts
custom_prompt_template = "Keep your answers limited to one word while answering this question: {query}"

# Function to stream responses
def stream_query_response(query):
    custom_prompt = custom_prompt_template.format(query=query)
    final_response = ""

    # Stream the response from the agent
    for event in agent_executor_with_memory.stream(
        {"messages": [HumanMessage(content=custom_prompt)]},
        config=config,
        stream_mode="values",
    ):
        final_response = event
        yield final_response

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("LangChain Chatbot with Streamlit Frontend")

# Display chat history
for chat in st.session_state.chat_history:
    with st.container():
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

# User input
user_input = st.text_input("You:", key="input")

if st.button("Send") and user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"user": user_input, "bot": ""})
    latest_index = len(st.session_state.chat_history) - 1

    # Placeholder for bot response
    response_placeholder = st.empty()

    # Stream the response
    for response in stream_query_response(user_input):
        st.session_state.chat_history[latest_index]['bot'] = response
        # Update the placeholder with the latest response
        response_placeholder.markdown(f"**Bot:** {response}")
