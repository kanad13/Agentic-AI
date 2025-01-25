import streamlit as st  # For creating web applications
from dotenv import load_dotenv  # For loading environment variables from a .env file
import os  # For interacting with the operating system, e.g., environment variables
import uuid  # For generating unique identifiers
from langchain_openai import ChatOpenAI  # OpenAI's chat model integration for Langchain
from langchain_google_genai import ChatGoogleGenerativeAI  # Google's Gemini chat model integration for Langchain
from langchain_groq import ChatGroq  # Groq's chat model integration for Langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader  # For loading PDF documents from a directory
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain_huggingface import HuggingFaceEmbeddings  # Hugging Face embeddings for text vectorization
from langchain_community.vectorstores import FAISS  # FAISS vector store for efficient similarity search
from langchain.tools.retriever import create_retriever_tool  # For creating tools from retrievers
from langchain_community.tools.tavily_search import TavilySearchResults  # Tavily Search tool for internet searches
from langgraph.checkpoint.memory import MemorySaver  # For saving and restoring agent states in memory
from langgraph.prebuilt import create_react_agent  # For creating ReAct agents in Langchain
from langchain_core.prompts import PromptTemplate  # For creating prompt templates
from langchain.schema import AIMessage, HumanMessage, SystemMessage # Message types for chat conversations
from langchain_community.retrievers import WikipediaRetriever # Retriever for Wikipedia articles
import logging # For logging events and errors


# Load environment variables from .env file
# This allows storing sensitive information like API keys outside of the code.
load_dotenv()
logging.basicConfig(level=logging.ERROR) # Configure basic logging to capture errors

############§§§§§§§§§§§§§§§§§§§§§############

# Streamlit Page Configuration
st.set_page_config(
    page_title="ChatBot", # Title of the Streamlit app in the browser tab
    page_icon=":chat-plus-outline:", # Icon for the Streamlit app in the browser tab
    layout="wide", # Use a wide layout for the app
    initial_sidebar_state="expanded", # Initially expand the sidebar
    menu_items=None # Remove the default Streamlit menu
)

# Retrieve API keys from environment variables
# These keys are used to authenticate with different AI models and services.
GROQ_API_KEY = os.getenv('GROQ_API_KEY') # API key for Groq models
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') # API key for Google models
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # API key for OpenAI models
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY') # API key for Tavily Search
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY') # API key for Langchain Observability/Tracing features
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2') # Flag to enable Langchain Tracing V2 for debugging and monitoring

############§§§§§§§§§§§§§§§§§§§§§############

# Model Selection in Sidebar
st.sidebar.title("Settings") # Title for the sidebar settings section
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=["gpt-4o-mini", "gemini-2.0-flash-exp", "gemma2-9b-it"] # Dropdown to choose the chatbot model
)

# Initialize Chat Model based on user selection
if selected_model == "gpt-4o-mini":
    model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY) # Use OpenAI's GPT-4o model
elif selected_model == "gemini-2.0-flash-exp":
    model = ChatGoogleGenerativeAI(model ="gemini-2.0-flash-exp", google_api_key=GOOGLE_API_KEY) # Use Google's Gemini model
elif selected_model == "gemma2-9b-it":
    model = ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY) # Use Groq's Gemma model


# Configure Tokenizers Parallelism and Embedding Batch Size
os.environ["TOKENIZERS_PARALLELISM"] = "true" # Enable parallel tokenization for potentially faster processing
embedding_batch_size = 512 # Set batch size for embedding operations; adjust based on available memory and GPU (if applicable). Larger batch size can be faster but requires more memory.

############§§§§§§§§§§§§§§§§§§§§§############

# Document Loading Function
@st.cache_resource # Cache the output of this function to avoid reloading documents on every run
def load_documents():
    """Loads PDF documents from the './input_files/' directory."""
    loader = PyPDFDirectoryLoader(path="./input_files/") # Load PDF documents from the specified directory
    return loader.load() # Return the loaded documents

loaded_documents = load_documents() # Load documents into memory

# Document Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Maximum size of each text chunk in characters
    chunk_overlap=200, # Number of overlapping characters between adjacent chunks to maintain context
    add_start_index=True # Include the starting index of each chunk in the original document (useful for source tracking)
)

document_chunks = text_splitter.split_documents(loaded_documents) # Split loaded documents into chunks

# Vector Store Creation Function
@st.cache_resource # Cache the output of this function to avoid recreating the vector store on every run
def create_vectorstore():
    """Creates and caches a FAISS vector store from document chunks using HuggingFace embeddings."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2", # Pre-trained sentence embedding model from Hugging Face
        encode_kwargs={'batch_size': embedding_batch_size} # Batch size for embedding generation
    )
    return FAISS.from_documents(documents=document_chunks, embedding=embedding_model) # Create FAISS vector store from document chunks and embeddings

vectorstore = create_vectorstore() # Create and load the vector store

# Document Retriever
document_retriever = vectorstore.as_retriever(
    search_type="similarity", # Use similarity search to find relevant documents
    search_kwargs={"k": 6} # Retrieve the top 6 most similar document chunks
)

############§§§§§§§§§§§§§§§§§§§§§############

# Tool Definitions for Agent
# Tools enhance the agent's capabilities by providing access to external resources and functionalities.

# Document Retrieval Tool
retriever_tool = create_retriever_tool(
    document_retriever, # The retriever object to use for document retrieval
    "retriever_tool", # Name of the tool, used by the agent to invoke it
    "Retrieves and provides information from the input documents." # Description of the tool, used by the agent to understand its purpose
)

# Internet Search Tool (Tavily)
internet_search_tool = TavilySearchResults(
    max_results=2, # Limit the number of internet search results to reduce noise and processing time
    search_depth="advanced", # Perform a more thorough and in-depth internet search
    include_answer=True, # Include a direct answer from the search results if available
    include_raw_content=True, # Include the raw content of the search results for more detailed information
    include_images=True, # Allow the search results to include images
)

# Wikipedia Retrieval Tool
wikipedia_retriever = WikipediaRetriever() # Initialize Wikipedia retriever
wikipedia_retriever_tool = create_retriever_tool(
    wikipedia_retriever, # The Wikipedia retriever object
    "wikipedia_retriever_tool", # Name of the Wikipedia retrieval tool
    "Retrieves and provides information from Wikipedia articles." # Description of the tool
)

# List of Tools for the Agent
tools = [retriever_tool, wikipedia_retriever_tool, internet_search_tool] # Combine all defined tools into a list

############§§§§§§§§§§§§§§§§§§§§§############

# Memory Setup for Conversation History
memory = MemorySaver() # Initialize MemorySaver to persist conversation history

# Unique Thread ID for Conversation Management
if 'unique_id' not in st.session_state:
    st.session_state.unique_id = str(uuid.uuid4()) # Generate a unique ID if not already present in session state
agent_config = {"configurable": {"thread_id": st.session_state.unique_id}} # Configuration for the agent, including the unique thread ID

# Agent Creation with Memory
agent_with_memory = create_react_agent(model, tools, checkpointer=memory) # Create a ReAct agent with the selected model, tools, and memory saver

# Custom Prompt Template for Agent
custom_prompt_template = PromptTemplate(
    template="""
    You are an AI assistant equipped with the following tools:

    ### Tool Descriptions: ###
    - **retriever_tool**: This is a Retrieval Augmented Generation Tool that retrieves information from input documents.
    - **wikipedia_retriever_tool**: Retrieves information from Wikipedia articles.
    - **internet_search_tool**: Conducts real-time internet searches for current information using Tavily Search.

    ### Instructions for Tool Usage: ###
    When answering queries posed by the user, follow this sequence of steps to ensure comprehensive and accurate responses:
    1. **Document Retrieval**: First, check for an answer to the user's query by invoking the `retriever_tool`. This tool searches the provided input documents for relevant information.
    2. **Wikipedia Search**: If no answer is found within the information retrieved by the `retriever_tool`, then invoke the `wikipedia_retriever_tool` to seek an answer from Wikipedia articles.
    3. **Internet Search**: If no answer is found in either the input documents or Wikipedia, then invoke the `internet_search_tool` to conduct a broader internet search for the answer.

    ### Fallback Response: ###
    If, after checking all these sources, the retrieved information still does not address the user's question, respond with:
    "I'm sorry, but I don't have the information to answer that question."

    ### Important Constraints: ###
    It is crucial to avoid fabricating or speculating on information. Do not generate content beyond the retrieved data. Always cite your source for information when possible, e.g., mention the name of the input document that was used to retrieve the information or indicate if the information is from Wikipedia or the internet search.

    Here is the input from the user: {query}
    """,
    input_variables=["query"] # Input variable that the prompt template expects
)

############§§§§§§§§§§§§§§§§§§§§§############

# Function to Stream Chatbot Responses
def stream_query_response(query, debug_mode=False, show_event_data=False, show_tool_calls=False):
    """
    Streams responses from the chatbot agent based on the user query.

    Args:
        query (str): The user's input query.
        debug_mode (bool, optional): Enables debug logging to display detailed information. Defaults to False.
        show_event_data (bool, optional): Shows raw event data from the agent stream in an expander. Defaults to False.
        show_tool_calls (bool, optional): Shows details of tool calls made by the agent in an expander. Defaults to False.

    Yields:
        str: Partial responses from the chatbot, streamed as they are generated.
    """
    # Initialize conversation history with the system prompt
    previous_messages = [SystemMessage(content=custom_prompt_template.format(query=query))]

    # Append previous chat messages from session state to maintain conversation context
    for chat in st.session_state.chat_history:
        previous_messages.append(HumanMessage(content=chat['user'])) # Add user message from history
        if chat['bot']:
            previous_messages.append(AIMessage(content=chat['bot'])) # Add bot response from history

    # Add the current user query to the message history
    previous_messages.append(HumanMessage(content=query))

    full_response = "" # Accumulate the full response from the stream
    text_output = "" # Accumulate debug text output
    tool_calls_output = "" # Accumulate tool calls output

    try:
        if debug_mode:
            text_output += "Debug Log:\n"
            text_output += "--------------------\n"
            text_output += "Initial Messages to Agent:\n"
            for msg in previous_messages:
                text_output += f"- {msg}\n"
            text_output += "\nAgent Stream Output:\n"
        # Stream responses from the agent
        for event in agent_with_memory.stream(
            {"messages": previous_messages}, config=agent_config, stream_mode="values" # Pass messages and agent config for streaming
        ):
            if isinstance(event, (str, dict)): # Handle string and dictionary events from the stream
                if isinstance(event, dict): # Process dictionary events (typically containing structured messages)
                    if event.get('messages'): # Check if the event is a message event (LangGraph agent may emit other types of events)
                        last_message = event['messages'][-1] # Get the latest message from the event
                        full_response = last_message.content # Extract the content of the last message

                        if debug_mode:
                            text_output += f"\n**Message Type**: {type(last_message).__name__}\n" # Log message type
                            text_output += f"**Content**: {last_message.content}\n" # Log message content
                            if isinstance(last_message, AIMessage) and last_message.tool_calls: # Check if it's an AI message and has tool calls
                                text_output += "**Tool Calls**:\n"
                                tool_calls_output += "**Tool Calls**:\n"
                                for tool_call in last_message.tool_calls:
                                    tool_call_debug_str = f"  - **Tool Name**: {tool_call['name']}\n"
                                    tool_call_debug_str += f"    **Tool Args**: {tool_call['args']}\n"
                                    text_output += tool_call_debug_str
                                    tool_calls_output += tool_call_debug_str
                else: # Handle string events (raw text responses)
                    full_response = event # Assign the string event as the full response
                    if debug_mode:
                        text_output += f"\n**String Event**: {event}\n" # Log string event
            elif debug_mode:
                text_output += f"\n**Event**: {event}\n" # Log other event types if in debug mode
            yield full_response # Yield the partial response to the Streamlit app
        st.session_state.chat_history[latest_index]['bot'] = full_response # Update chat history with the full bot response
        if debug_mode:
            st.session_state.debug_output = text_output # Store debug output in session state
        if show_tool_calls:
            st.session_state.tool_calls_output = tool_calls_output # Store tool calls output in session state
        if show_event_data: # Store the last event data in session state after the loop
            st.session_state.event_data = event # Store the last 'event'
    except Exception as e: # Error handling for response processing
        logging.error(f"Error processing response: {e}", exc_info=True) # Log the error
        yield "I encountered an error processing your request. Please try again later." # Yield an error message to the user
    # ... (rest of stream_query_response) ...


############§§§§§§§§§§§§§§§§§§§§§############
# Initialize Session State Variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # Initialize chat history list to store user and bot messages
if 'debug_output' not in st.session_state:
    st.session_state.debug_output = "" # Initialize debug output string
if 'tool_calls_output' not in st.session_state:
    st.session_state.tool_calls_output = "" # Initialize tool calls output string
if 'event_data' not in st.session_state: # Initialize event_data
    st.session_state.event_data = None # Initialize event_data to None or ""

# Streamlit App Title
st.title("LangChain Chatbot with Streamlit Frontend") # Set the title of the Streamlit application

############§§§§§§§§§§§§§§§§§§§§§############

# Sidebar Checkboxes and Help Section
with st.sidebar.expander("Help & Display Options",  expanded=True):
    show_tool_calls = st.checkbox("Show Tool Calls", value=True) # Checkbox to show tool call details
    st.caption("Display details of tools used by the chatbot to answer your query.") # Description for "Show Tool Calls"

    debug_mode = st.checkbox("Show Debug Log", value=False) # Checkbox to enable debug log display
    st.caption("Enable detailed technical logs for debugging and advanced understanding.") # Description for "Show Debug Log"

    show_event_data = st.checkbox("Show Event Data", value=False) # Checkbox to show raw event data
    st.caption("Show raw communication data from the chatbot agent (technical).") # Description for "Show Event Data"


# Display Chat History from Session State
for chat in st.session_state.chat_history: # Iterate through chat history
    with st.chat_message("user"): # Display user messages in chat format
        st.write(chat['user']) # Write user message
    if chat['bot']: # Check if there is a bot response
        with st.chat_message("assistant"): # Display bot messages in chat format
            st.write(chat['bot']) # Write bot message

# User Input Handling
if user_input := st.chat_input("You:"): # Get user input from the chat input box
    # Append user message to chat history
    st.session_state.chat_history.append({"user": user_input, "bot": ""}) # Add user message and empty bot response to chat history
    latest_index = len(st.session_state.chat_history) - 1 # Get the index of the latest message

    # Display User Message in Chat
    with st.chat_message("user"): # Display user message in chat format
        st.write(user_input) # Write user message

    # Placeholder for Bot Response
    with st.chat_message("assistant"): # Create a chat message container for the assistant
        response_placeholder = st.empty() # Create an empty placeholder within the assistant chat message
        full_response = ""  # Initialize an empty full response
        with st.spinner("Thinking..."): # Display a spinner while waiting for the response
          for response in stream_query_response(user_input, debug_mode=debug_mode, show_event_data=show_event_data, show_tool_calls=show_tool_calls): # Stream chatbot responses
            full_response = response  # Accumulate full response
            response_placeholder.markdown(full_response) # Update the placeholder with the accumulated response

# Conditional Display of Debug and Tool Call Output Expanders
if show_tool_calls: # Conditionally display tool calls output
  if st.session_state.tool_calls_output: # Check if there is tool calls output to display
    with st.expander("Tool Interaction Details"):
        st.write("This section reveals the tools the chatbot used to respond to your query. It shows which tools were activated and what instructions were given to them. This can help you understand how the chatbot is working behind the scenes to find information.")
        st.code(st.session_state.tool_calls_output) # Display tool calls output in an expander

if debug_mode: # Conditionally display debug output
  if st.session_state.debug_output: # Check if there is debug output to display
    with st.expander("Detailed Debugging Information"):
        st.write("This section provides a detailed technical log of the chatbot's thought process. It's useful for understanding exactly what steps the chatbot took to answer your question, including the messages sent back and forth internally. This level of detail is generally for debugging and advanced understanding.")
        st.code(st.session_state.debug_output) # Display debug output in an expander

if show_event_data: # Conditionally display event data expander
  if st.session_state.event_data: # Check if there is event data to display
    with st.expander("Raw Agent Communication Data (Technical)"):
        st.write("This section displays the raw, technical data stream from the chatbot agent. This is advanced debugging information showing the step-by-step communication within the agent as it processes your request. It's primarily useful for developers or those deeply interested in the technical workings.")
        st.write("Event Details:", st.session_state.event_data) # Display stored event data from session state
        # Note: Currently showing the *last* event received in the stream. Consider if you want to display all events or a summary.
