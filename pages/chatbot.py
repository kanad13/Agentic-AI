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
# from langchain_community.tools import WikipediaQueryRun # Removed
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
config = {
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


# This variable indicates whether the model supports streaming data processing.
# Streaming can be useful for handling large datasets or real-time data processing.
does_model_support_streaming = True
# Set an environment variable to control the behavior of tokenizers.
# This setting allows tokenizers to work in parallel, potentially speeding up text processing tasks.
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Define the batch size for embedding operations.
# This setting determines how many items are processed at once when creating embeddings, which can affect both memory usage and processing speed.
embedding_batch_size = 512

############§§§§§§§§§§§§§§§§§§§§§############

# Define a function to load documents from a directory

# This decorator caches the result of the function, which means:
# - The function's output will be stored in memory after the first call.
# - Subsequent calls will retrieve the cached result instead of re-executing the function, which can significantly speed up repeated operations.
@st.cache_resource
def load_documents():
    # This function is responsible for loading PDF documents from a specified directory.
    # It uses PyPDFDirectoryLoader to handle multiple PDF files at once.
     # The path "./input_files/" is where the PDF files are expected to be located.
    loader = PyPDFDirectoryLoader(path="./input_files/")

    # The `load()` method of the loader is called to actually load the documents into memory.
    # This step prepares the documents for further processing or analysis.
    return loader.load()

# Call the `load_documents()` function and store the result in the `docs` variable.
# This variable will now contain all the documents loaded from the directory,
# ready for use in subsequent parts of the script or notebook.
docs = load_documents()

# Split the loaded documents into smaller chunks for processing
# Create an instance of RecursiveCharacterTextSplitter to split documents into smaller chunks.
# This class is used to divide large documents into manageable pieces for processing:
# - `chunk_size`: Specifies the maximum number of characters in each chunk (1000 in this case).
# - `chunk_overlap`: Allows for some overlap between chunks to maintain context (200 characters).
# - `add_start_index`: Adds the starting index of each chunk in the original document for reference.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

# Use the `split_documents` method of the text_splitter to process all documents in `docs`.
# This method will:
# - Split each document into chunks based on the specified parameters.
# - Return a list of these chunks, which can then be used for further processing or analysis.
all_splits = text_splitter.split_documents(docs)

# Create a vector store for efficient document retrieval
# This decorator caches the result of the function, which means:
# - The function's output will be stored in memory after the first call.
# - Subsequent calls will retrieve the cached result instead of re-executing the function, which can significantly speed up repeated operations.
@st.cache_resource
def create_vectorstore():
    # This function sets up an embedding model and indexes the document chunks for retrieval:
    # - It uses the HuggingFaceEmbeddings class to create an embedding model.
    # - The model chosen here is 'sentence-transformers/all-mpnet-base-v2', known for its performance in sentence embeddings.
    # - The `encode_kwargs` parameter sets the batch size for embedding, which was defined earlier as `embedding_batch_size`.
    embedding_wrapper = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={'batch_size': embedding_batch_size}
    )
    # Use FAISS (Facebook AI Similarity Search) to create an index from the document chunks:
    # - FAISS is an efficient similarity search and clustering library.
    # - It takes the document chunks (`all_splits`) and the embedding model to create an index.
    # - This index allows for fast retrieval of similar documents or chunks based on their embeddings.
    return FAISS.from_documents(documents=all_splits, embedding=embedding_wrapper)

# Call the `create_vectorstore()` function and store the result in the `vectorstore` variable.
# This variable now contains an index of all document chunks, ready for similarity search or retrieval operations.
vectorstore = create_vectorstore()

# Create a retriever object from the vector store
# This object will be used to retrieve documents based on similarity search:
# - `search_type="similarity"` specifies that we want to find documents similar to a given query.
# - `search_kwargs={"k": 6}` means we want to retrieve the top 6 most similar documents.
retriever_object = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

############§§§§§§§§§§§§§§§§§§§§§############

# Define tools for the chatbot to use
# These tools provide different functionalities for information retrieval:

# Create a tool for retrieving information from the input documents:
# This tool uses the `retriever_object` to fetch relevant document chunks.
retriever_tool = create_retriever_tool(
    retriever_object,
    "retriever_tool",
    "Retrieves and provides information from the input documents.",
)

# Set up an internet search tool:
# - `max_results=2` limits the number of search results to 2.
# - `search_depth="advanced"` specifies a more thorough search.
# - `include_answer=True` includes a direct answer if available.
# - `include_raw_content=True` includes the raw content of the search results.
# - `include_images=True` allows for image results to be included.
internet_search_tool = TavilySearchResults(
    max_results=2,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

# WikipediaRetriever is designed for retrieving documents to be used in downstream tasks within a pipeline, while WikipediaQueryRun is intended for direct querying of Wikipedia, often within agent-based frameworks.

#wikipedia_search_tool = WikipediaQueryRun(
#    api_wrapper=WikipediaAPIWrapper()
#)

# WikipediaRetriever is used for retrieving documents from Wikipedia:
# - This tool fetches Wikipedia articles for further processing or analysis.
wikipedia_retriever = WikipediaRetriever()
wikipedia_retriever_tool = create_retriever_tool(
    wikipedia_retriever,
    "wikipedia_retriever_tool",
    "Retrieves and provides information from Wikipedia articles.",
)

# List of tools available to the chatbot:
# This list contains the tools that the chatbot can use to answer queries.
tools = [retriever_tool, wikipedia_retriever_tool, internet_search_tool]
#tools = [retriever_tool, wikipedia_search_tool, internet_search_tool]

############§§§§§§§§§§§§§§§§§§§§§############

# Setup memory for conversation history
# Initialize a MemorySaver object, which will be used to manage and store conversation history.
# MemorySaver is part of LangGraph's checkpointing system, allowing for persistence of conversation state.
memory = MemorySaver()

# Generate a unique ID for each thread to manage conversation history
# Persist the unique thread ID in `st.session_state` to maintain consistency across interactions.
if 'unique_id' not in st.session_state:
    st.session_state.unique_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": st.session_state.unique_id}}

# Create an agent with memory capabilities
# Creating a ReAct agent with memory capabilities:
# - `model` is the language model used for generating responses.
# - `tools` are the tools defined earlier for information retrieval.
# - `checkpointer=memory` integrates the MemorySaver into the agent, allowing it to save and load conversation history.
agent_executor_with_memory = create_react_agent(model, tools, checkpointer=memory)

# Define a custom prompt template for the chatbot
# This custom prompt template will guide the chatbot's behavior:
# - It provides instructions on how to respond to user queries.
# - It includes placeholders for dynamic content like the user's question or previous conversation history.
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
def stream_query_response(query, debug_mode=False, show_event_data=False): # Added show_event_data parameter
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
    text_output = "" # Initialize text output string for debug

    try:
        if debug_mode:
            text_output += "Debug Log:\n"
            text_output += "--------------------\n"
            text_output += "Initial Messages to Agent:\n"
            for msg in previous_messages:
                text_output += f"- {msg}\n"
            text_output += "\nAgent Stream Output:\n"
        # Stream the response from the agent with full message history
        for event in agent_executor_with_memory.stream(
            {"messages": previous_messages}, config=config, stream_mode="values"
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
                                    text_output += f"  - **Tool Name**: {tool_call['name']}\n" # Corrected line
                                    text_output += f"    **Tool Args**: {tool_call['args']}\n" # Corrected line
                 else:
                      full_response = event
                      if debug_mode:
                          text_output += f"\n**String Event**: {event}\n" # Handling simple string events if any
            elif debug_mode:
                text_output += f"\n**Event**: {event}\n" # For other event types if needed
            yield full_response
        if show_event_data: # Conditional display of Event Data expander
            with st.expander("Show Event Data"): # Keep original expander for raw event
                st.write("Event Details:", event)
    except Exception as e:
        logging.error(f"Error processing response: {e}", exc_info=True)
        yield "I encountered an error processing your request. Please try again later."
    st.session_state.chat_history[latest_index]['bot'] = full_response
    if debug_mode: # Display the formatted text output in debug mode
        st.session_state.debug_output = text_output # Store for display in sidebar

############§§§§§§§§§§§§§§§§§§§§§############
# Initialize session state for chat history
# This block checks if 'chat_history' exists in the session state. If not, it initializes an empty list.
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'debug_output' not in st.session_state: # Initialize debug output in session state
    st.session_state.debug_output = ""

# Display chat history
# Set the title for the Streamlit app, which will be the header for the chatbot interface.
st.title("LangChain Chatbot with Streamlit Frontend")

############§§§§§§§§§§§§§§§§§§§§§############

# Debug mode toggle in the sidebar
# This section creates a sidebar in the Streamlit app for settings:
# - A title for the settings section is added.
# - A checkbox allows users to toggle debug mode on or off, which can be useful for development or troubleshooting.
#st.sidebar.title("Settings") # Moved to the model selection above
debug_mode = st.sidebar.checkbox("Show Debug Log", value=False) # Checkbox for Debug Log
show_event_data = st.sidebar.checkbox("Show Event Data", value=False) # Checkbox for Event Data

# Display chat history
# This loop iterates through the chat history stored in the session state:
# - For each chat entry, it displays the user's message in a 'user' chat message container.
# - If there's a bot response, it displays that in an 'assistant' chat message container.
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat['user'])
    if chat['bot']:
        with st.chat_message("assistant"):
            st.write(chat['bot'])

# User input
# This block handles user input through Streamlit's chat input widget:
# - `st.chat_input` creates an input field for the user to type their message.
# - The `:=` operator assigns the input to `user_input` if it's not empty.
if user_input := st.chat_input("You:"):
    # Add user message to chat history
    # Append the user's input to the chat history, initializing the bot's response as an empty string.
    st.session_state.chat_history.append({"user": user_input, "bot": ""})
    latest_index = len(st.session_state.chat_history) - 1

    # Display the user's input
    # Show the user's message in a 'user' chat message container for visual feedback.
    with st.chat_message("user"):
        st.write(user_input)

    # Placeholder for bot response
    # Create a placeholder for the bot's response, which will be updated in real-time.
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

    # Stream the response
    # This loop streams the chatbot's response:
    # - `stream_query_response` is called with the user's input, debug mode, and event data settings.
    # - Each response chunk is processed as it's generated.
    full_response = ""
    for response in stream_query_response(user_input, debug_mode=debug_mode, show_event_data=show_event_data): # Pass show_event_data
        full_response = response
        response_placeholder.markdown(response)

# Display debug output in main window if debug_mode is enabled
# Display Debug Log after the chatbot response and when debug mode is enabled
if debug_mode:
  if st.session_state.debug_output: # Check if there is debug output to show
    st.expander("Show Debug Log").code(st.session_state.debug_output)
# Event Data expander is now conditionally displayed inside stream_query_response based on show_event_data checkbox
