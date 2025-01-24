# Import necessary libraries for Streamlit, environment variables, and other utilities
import streamlit as st
from dotenv import load_dotenv
import os
import uuid
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.tools import WikipediaQueryRun  # Removed: Replaced by WikipediaRetriever tool
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.retrievers import WikipediaRetriever
import logging
import json


# Load environment variables from a .env file
load_dotenv()
logging.basicConfig(level=logging.ERROR)

############§§§§§§§§§§§§§§§§§§§§§############

# Configure Streamlit page settings
streamlit_page_config = {
    'scrollZoom': True,
    'displayModeBar': True,
    'displaylogo': False
}
st.set_page_config(
    page_title="ChatBot",
    page_icon=":chat-plus-outline:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Retrieve API keys from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')

############§§§§§§§§§§§§§§§§§§§§§############

# Define the model for the chatbot
# Model Selection in Streamlit
st.sidebar.title("Settings")
selected_model = st.sidebar.selectbox("Select Model", options=["gpt-4o-mini", "gemini-2.0-flash-exp", "gemma2-9b-it"])

if selected_model == "gpt-4o-mini":
    model = ChatOpenAI(model="gpt-4o-mini")
elif selected_model == "gemini-2.0-flash-exp":
    model = ChatGoogleGenerativeAI(model ="gemini-2.0-flash-exp", google_api_key=GOOGLE_API_KEY)
elif selected_model == "gemma2-9b-it":
    model = ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY)


# Enable parallel tokenization for potential speed improvement.
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Batch size for embedding operations (adjust based on memory and speed).
embedding_batch_size = 512

############§§§§§§§§§§§§§§§§§§§§§############

# Define a function to load documents from a directory
# Cache loaded documents to avoid reloading on each run.
@st.cache_resource
def load_documents():
    # Loads PDF documents from the "./input_files/" directory.
    loader = PyPDFDirectoryLoader(path="./input_files/")
    return loader.load()

loaded_documents = load_documents()

# Split the loaded documents into smaller chunks for processing
# Initialize text splitter to chunk documents for processing (chunk_size=1000, chunk_overlap=200).
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

document_chunks = text_splitter.split_documents(loaded_documents)

# Create a vector store for efficient document retrieval
# Cache vector store to avoid re-creation on each run.
@st.cache_resource
def create_vectorstore():
    # Creates and caches a vector store from document chunks.
    # Uses 'sentence-transformers/all-mpnet-base-v2' for embeddings.
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={'batch_size': embedding_batch_size}
    )
    return FAISS.from_documents(documents=document_chunks, embedding=embedding_model)

vectorstore = create_vectorstore()

# Create a retriever object from the vector store
# Create retriever to fetch top 6 similar document chunks.
document_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

############§§§§§§§§§§§§§§§§§§§§§############

# Define tools for the chatbot to use
# These tools provide different functionalities for information retrieval.

# Create a tool for retrieving information from the input documents.
retriever_tool = create_retriever_tool(
    document_retriever,
    "retriever_tool",
    "Retrieves and provides information from the input documents.",
)

# Set up an internet search tool.
internet_search_tool = TavilySearchResults(
    max_results=2, # Limit internet search results to 2.
    search_depth="advanced", # Specify a more thorough search.
    include_answer=True, # Include a direct answer if available.
    include_raw_content=True, # Include the raw content of the search results.
    include_images=True, # Allow for image results to be included.
)


wikipedia_retriever = WikipediaRetriever()
wikipedia_retriever_tool = create_retriever_tool(
    wikipedia_retriever,
    "wikipedia_retriever_tool",
    "Retrieves and provides information from Wikipedia articles.",
)

# List of tools available to the chatbot.
tools = [retriever_tool, wikipedia_retriever_tool, internet_search_tool]

############§§§§§§§§§§§§§§§§§§§§§############

# Setup memory for conversation history
# Initialize MemorySaver for conversation history persistence.
memory = MemorySaver()

# Generate a unique ID for each thread to manage conversation history
# Persist the unique thread ID in `st.session_state` to maintain consistency across interactions.
if 'unique_id' not in st.session_state:
    st.session_state.unique_id = str(uuid.uuid4())
agent_config = {"configurable": {"thread_id": st.session_state.unique_id}}

# Create an agent with memory capabilities
# Create ReAct agent with memory using specified model, tools, and memory saver.
agent_with_memory = create_react_agent(model, tools, checkpointer=memory)

# Define a custom prompt template for the chatbot
# This custom prompt template will guide the chatbot's behavior.
custom_prompt_template = PromptTemplate(
    template="""
    You are an AI assistant equipped with the following tools:
    - **retriever_tool**: This is a Retrieval Augmented Generation Tool that retrieves information from input documents.
    - **wikipedia_retriever_tool**: Retrieves information from Wikipedia articles.
    - **internet_search_tool**: Conducts real-time internet searches for current information using Tavily Search.

    When answering queries posed by the user, follow this sequence:
    1. First check for answer to user's query by invoking the retriever_tool
    2. If no answer is found within the information retrieved by the retriever_tool, then invoke the wikipedia_retriever_tool and seek answer to the user's query.
    3. If no answer is found within the information retrieved by the retriever_tool and then the wikipedia_retriever_tool, then invoke the internet_search_tool and seek answer to the user's query.

    If the retrieved information from any of these tools does not address the user's question, respond with:
    "I'm sorry, but I don't have the information to answer that question."

    Avoid fabricating or speculating on information. Do not generate content beyond the retrieved data. Always cite your source for information e.g., the name of the input document that was used to retrieve the information.

    Here is the input from the user: {query}
    """,
    input_variables=["query"]
)

############§§§§§§§§§§§§§§§§§§§§§############

# Define a function to stream responses from the chatbot
def stream_query_response(query, debug_mode=False, show_event_data=False, show_tool_calls=False):
    # Initialize previous messages with the custom prompt as a system message
    previous_messages = [SystemMessage(content=custom_prompt_template.format(query=query))]

    # Append the chat history
    for chat in st.session_state.chat_history:
        previous_messages.append(HumanMessage(content=chat['user']))
        if chat['bot']:
            previous_messages.append(AIMessage(content=chat['bot']))

    # Add current query
    previous_messages.append(HumanMessage(content=query))

    full_response = ""
    text_output = ""
    tool_calls_output = ""

    try:
        if debug_mode:
            text_output += "Debug Log:\n"
            text_output += "--------------------\n"
            text_output += "Initial Messages to Agent:\n"
            for msg in previous_messages:
                text_output += f"- {msg}\n"
            text_output += "\nAgent Stream Output:\n"
        # Stream the response from the agent with full message history
        for event in agent_with_memory.stream(
            {"messages": previous_messages}, config=agent_config, stream_mode="values"
        ):
            if isinstance(event, (str, dict)):
                if isinstance(event, dict):
                    if event.get('messages'):
                        last_message = event['messages'][-1]
                        full_response = last_message.content

                        if debug_mode:
                            text_output += f"\n**Message Type**: {type(last_message).__name__}\n"
                            text_output += f"**Content**: {last_message.content}\n"
                            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                                text_output += "**Tool Calls**:\n"
                                for tool_call in last_message.tool_calls:
                                    text_output += f"  - **Tool Name**: {tool_call['name']}\n"
                                    text_output += f"    **Tool Args**: {tool_call['args']}\n"

                                    tool_calls_output += "**Tool Calls**:\n"
                                    for tool_call in last_message.tool_calls:
                                        tool_calls_output += f"  - **Tool Name**: {tool_call['name']}\n"
                                        tool_calls_output += f"    **Tool Args**: {tool_call['args']}\n"

                else:
                    full_response = event
                    if debug_mode:
                        text_output += f"\n**String Event**: {event}\n"
            elif debug_mode:
                text_output += f"\n**Event**: {event}\n"
            yield full_response
        if show_event_data:
            with st.expander("Show Event Data"):
                st.write("Event Details:", event)
    except Exception as e:
        logging.error(f"Error processing response: {e}", exc_info=True)
        yield "I encountered an error processing your request. Please try again later."
    st.session_state.chat_history[latest_index]['bot'] = full_response
    if debug_mode:
        st.session_state.debug_output = text_output
    if show_tool_calls:
        st.session_state.tool_calls_output = tool_calls_output
# Event Data expander is now conditionally displayed inside stream_query_response based on show_event_data checkbox

############§§§§§§§§§§§§§§§§§§§§§############
# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'debug_output' not in st.session_state:
    st.session_state.debug_output = ""
if 'tool_calls_output' not in st.session_state:
    st.session_state.tool_calls_output = ""

# Display chat history
# Set the title for the Streamlit app.
st.title("LangChain Chatbot with Streamlit Frontend")

############§§§§§§§§§§§§§§§§§§§§§############

# Debug mode toggle in the sidebar
# Sidebar checkboxes for debug and event/tool call details.
debug_mode = st.sidebar.checkbox("Show Debug Log", value=False)
show_event_data = st.sidebar.checkbox("Show Event Data", value=False)
show_tool_calls = st.sidebar.checkbox("Show Tool Calls", value=False)

# Display chat history from session state.
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat['user'])
    if chat['bot']:
        with st.chat_message("assistant"):
            st.write(chat['bot'])

# User input
# Handle user input via Streamlit chat input.
if user_input := st.chat_input("You:"):
    # Add user message to chat history
    st.session_state.chat_history.append({"user": user_input, "bot": ""})
    latest_index = len(st.session_state.chat_history) - 1

    # Display the user's input
    with st.chat_message("user"):
        st.write(user_input)

    # Placeholder for bot response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

    # Stream chatbot response and update placeholder.
    full_response = ""
    for response in stream_query_response(user_input, debug_mode=debug_mode, show_event_data=show_event_data, show_tool_calls=show_tool_calls):
        full_response = response
        response_placeholder.markdown(response)

# Conditionally display debug and tool call output expanders.
if debug_mode:
  if st.session_state.debug_output:
    st.expander("Show Debug Log").code(st.session_state.debug_output)

if show_tool_calls:
  if st.session_state.tool_calls_output:
    st.expander("Show Tool Calls").code(st.session_state.tool_calls_output)
