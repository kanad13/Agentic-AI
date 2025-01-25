# --- Core Libraries ---
import streamlit as st  # For creating web applications
import os  # For interacting with the operating system, e.g., environment variables
import uuid  # For generating unique identifiers
import logging # For logging events and errors
from dotenv import load_dotenv  # For loading environment variables from a .env file

# --- Langchain Framework ---
from langchain_openai import ChatOpenAI  # OpenAI's chat model integration for Langchain
from langchain_google_genai import ChatGoogleGenerativeAI  # Google's Gemini chat model integration for Langchain
from langchain_groq import ChatGroq  # Groq's chat model integration for Langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader  # For loading PDF documents from a directory
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain_huggingface import HuggingFaceEmbeddings  # Hugging Face embeddings for text vectorization
from langchain_community.vectorstores import FAISS  # FAISS vector store for efficient similarity search
from langchain.tools.retriever import create_retriever_tool  # For creating tools from retrievers
from langchain_community.tools.tavily_search import TavilySearchResults  # Tavily Search tool for internet searches
from langchain_community.retrievers import WikipediaRetriever # Retriever for Wikipedia articles
from langgraph.checkpoint.memory import MemorySaver  # For saving and restoring agent states in memory
from langgraph.prebuilt import create_react_agent  # For creating ReAct agents in Langchain
from langchain_core.prompts import PromptTemplate  # For creating prompt templates
from langchain.schema import AIMessage, HumanMessage, SystemMessage # Message types for chat conversations

# --- Overall Comment ---
# Import necessary libraries for chatbot functionality, including Streamlit for UI,
# Langchain for LLM interactions, vector database management, and utility libraries.


# --- Environment Variables ---
# Load environment variables from .env file.
# This is crucial for securely managing API keys and other sensitive configurations outside of the main codebase, improving security and deployment flexibility.
load_dotenv()

# --- Basic Error Logging ---
logging.basicConfig(level=logging.ERROR) # Configure basic logging to capture errors.
# Setting the logging level to ERROR ensures that only error messages and above (critical, fatal) are captured. This is useful for debugging and monitoring the application for critical issues.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="ChatBot", # Title of the Streamlit app in the browser tab
    page_icon=":chat-plus-outline:", # Icon for the Streamlit app in the browser tab
    layout="wide", # Use a wide layout for the app
    initial_sidebar_state="expanded", # Initially expand the sidebar
    menu_items=None # Remove the default Streamlit menu
)

# --- API Key Retrieval ---
# Retrieve API keys from environment variables.
# These keys are essential for authenticating with various AI models and services.
# Ensure these API keys are correctly set in your .env file.
# Refer to the respective provider's documentation for obtaining these keys.

GROQ_API_KEY = os.getenv('GROQ_API_KEY') # API key for Groq models. Required for accessing Groq's language models.
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') # API key for Google models. Required for accessing Google's Gemini language models.
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # API key for OpenAI models. Required for accessing OpenAI's language models like GPT-4o.
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY') # API key for Tavily Search. Required for using the Tavily internet search tool.
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY') # API key for Langchain Observability/Tracing features. Enables advanced debugging and monitoring of Langchain applications.
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2') # Flag to enable Langchain Tracing V2 for more detailed observability. Set to 'true' to activate. Useful for debugging and performance analysis.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Model Selection in Sidebar ---
st.sidebar.title("Settings") # Title for the sidebar settings section
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=["gpt-4o-mini", "gemini-2.0-flash-exp", "gemma2-9b-it"] # Dropdown to choose the chatbot model.
    # Model options are selected for a balance of performance and cost.
    # 'gpt-4o-mini' represents OpenAI's latest model, 'gemini-2.0-flash-exp' is Google's fast model,
    # and 'gemma2-9b-it' is a powerful open-source model from Groq.
)

# Initialize Chat Model based on user selection
if selected_model == "gpt-4o-mini":
    model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY) # Use OpenAI's GPT-4o model. Integrates with OpenAI's API using the selected model.
elif selected_model == "gemini-2.0-flash-exp":
    model = ChatGoogleGenerativeAI(model ="gemini-2.0-flash-exp", google_api_key=GOOGLE_API_KEY) # Use Google's Gemini model. Integrates with Google's Gemini API.
elif selected_model == "gemma2-9b-it":
    model = ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY) # Use Groq's Gemma model. Integrates with Groq's API.


# --- Performance Tuning ---
# Configure Tokenizers Parallelism and Embedding Batch Size
os.environ["TOKENIZERS_PARALLELISM"] = "true" # Enable parallel tokenization for potentially faster processing.
# This can speed up text processing, especially when dealing with large documents, by utilizing multiple CPU cores.
# However, it might slightly increase resource consumption.

embedding_batch_size = 512 # Set batch size for embedding operations. Adjust based on available memory and GPU (if applicable).
# Larger batch sizes can improve embedding generation speed, but require more memory.
# A value of 512 is a reasonable starting point, but you may need to decrease it if you encounter memory issues,
# especially on systems with limited RAM or when processing very large documents.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Document Loading ---
# Document Loading Function
@st.cache_resource # Cache the output of this function to avoid reloading documents on every run.
# This decorator from Streamlit efficiently caches the result of 'load_documents'.
# Subsequent runs of the app will reuse the loaded documents unless the function's inputs change.
def load_documents():
    """Loads PDF documents from the './input_files/' directory."""
    loader = PyPDFDirectoryLoader(path="./input_files/") # Load PDF documents from the specified directory using Langchain's PyPDFDirectoryLoader.
    return loader.load() # Return the list of loaded documents.

loaded_documents = load_documents() # Load documents into memory. This will be cached for efficiency.

# --- Document Chunking ---
# Document Chunking
# Splitting documents into smaller chunks is essential for several reasons:
# 1. Language models have token limits (context window). Processing large documents directly is often not feasible.
# 2. Smaller chunks improve the relevance of document retrieval. Searching for relevant information is more effective within smaller, focused segments of text.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Maximum size of each text chunk in characters. This determines the length of each text segment.
    # A chunk size of 1000 characters is a common starting point, balancing context and processing efficiency.
    chunk_overlap=200, # Number of overlapping characters between adjacent chunks to maintain context continuity.
    # Overlap helps ensure that context isn't lost between chunks, especially when sentences or important phrases span chunk boundaries.
    add_start_index=True # Include the starting index of each chunk in the original document.
    # This is useful for tracing back the source of information and providing accurate citations.
)

document_chunks = text_splitter.split_documents(loaded_documents) # Split loaded documents into smaller, manageable chunks.

# --- Vector Store Creation ---
# Vector Store Creation Function
@st.cache_resource # Cache the output of this function to avoid recreating the vector store on every run.
# Caching is crucial for performance as vector store creation can be computationally expensive.
def create_vectorstore():
    """Creates and caches a FAISS vector store from document chunks using HuggingFace embeddings."""
    # --- Embedding Model ---
    # HuggingFace Embeddings are used to convert text chunks into numerical vectors.
    # These vectors capture the semantic meaning of the text, allowing for similarity search.
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2", # Pre-trained sentence embedding model from Hugging Face.
        # 'sentence-transformers/all-mpnet-base-v2' is a widely used and effective sentence embedding model
        # that provides good performance for general-purpose text similarity tasks.
        encode_kwargs={'batch_size': embedding_batch_size} # Batch size for embedding generation. Uses the globally defined batch size.
    )
    # --- FAISS Vector Store ---
    # FAISS (Facebook AI Similarity Search) is used as the vector database.
    # FAISS is highly efficient for similarity search in high-dimensional spaces, making it suitable for RAG applications.
    return FAISS.from_documents(documents=document_chunks, embedding=embedding_model) # Create FAISS vector store from document chunks and embeddings.

vectorstore = create_vectorstore() # Create and load the vector store. This is cached.

# --- Document Retriever ---
# Document Retriever
# The retriever is responsible for fetching relevant document chunks from the vector store based on a query.
document_retriever = vectorstore.as_retriever(
    search_type="similarity", # Use similarity search to find relevant documents.
    # 'similarity' search finds document chunks that are semantically similar to the query vector.
    search_kwargs={"k": 6} # Retrieve the top 6 most similar document chunks.
    # 'k=6' means the retriever will return the 6 most relevant document chunks for each query.
    # This number can be adjusted to balance precision and recall in document retrieval.
)

############§§§§§§§§§§§§§§§§§§§§§############

# --- Tool Definitions for Agent ---
# Tools enhance the agent's capabilities, allowing it to interact with the outside world
# and access information beyond its internal knowledge base.
# This is crucial for creating a more versatile and informative chatbot.

# Document Retrieval Tool
# This tool allows the agent to retrieve information from the loaded documents (PDF files).
retriever_tool = create_retriever_tool(
    document_retriever, # The retriever object created earlier, responsible for fetching document chunks.
    "retriever_tool", # Name of the tool, used by the agent to invoke it. This name is referenced in the agent's prompt.
    "Retrieves and provides information from the input documents." # Description of the tool, used by the agent to understand its purpose and when to use it.
)

# Internet Search Tool (Tavily)
# This tool provides the agent with the ability to search the internet for up-to-date information.
# It uses the Tavily Search API for real-time web search results.
internet_search_tool = TavilySearchResults(
    max_results=2, # Limit the number of internet search results to reduce noise and processing time.
    # Limiting results to 2 helps focus on the most relevant information and prevents overwhelming the agent.
    search_depth="advanced", # Perform a more thorough and in-depth internet search.
    # 'advanced' search depth aims to provide more comprehensive results, but might take slightly longer.
    include_answer=True, # Include a direct answer from the search results if available.
    # This can provide a concise answer directly from the search snippet if Tavily can extract one.
    include_raw_content=True, # Include the raw content of the search results for more detailed information.
    # Including raw content allows the agent to access the full text of the search results if needed.
    include_images=True, # Allow the search results to include images. While currently not directly used, this option is enabled for potential future features.
)

# Wikipedia Retrieval Tool
# This tool allows the agent to retrieve information from Wikipedia articles.
wikipedia_retriever = WikipediaRetriever() # Initialize Wikipedia retriever using Langchain's built-in Wikipedia retriever.
wikipedia_retriever_tool = create_retriever_tool(
    wikipedia_retriever, # The Wikipedia retriever object.
    "wikipedia_retriever_tool", # Name of the Wikipedia retrieval tool.
    "Retrieves and provides information from Wikipedia articles." # Description of the tool.
)

# List of Tools for the Agent
# Combine all defined tools into a list. This list will be passed to the agent creation function.
tools = [retriever_tool, wikipedia_retriever_tool, internet_search_tool] # List of tools available to the agent.
# The agent will decide which tool to use based on the user's query and the tool descriptions.
# The order in this list doesn't typically matter for functionality but can be organized logically.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Memory Setup for Conversation History ---
# Memory is crucial for creating conversational chatbots that can remember past interactions.
# MemorySaver is used here to persist conversation history, allowing the chatbot to maintain context across multiple turns.
memory = MemorySaver() # Initialize MemorySaver to persist conversation history using Langchain's MemorySaver.

# --- Unique Thread ID for Conversation Management ---
# A unique thread ID is generated for each user session to isolate conversations and manage state independently.
if 'unique_id' not in st.session_state:
    st.session_state.unique_id = str(uuid.uuid4()) # Generate a unique ID if not already present in session state using UUID.
agent_config = {"configurable": {"thread_id": st.session_state.unique_id}} # Configuration for the agent, including the unique thread ID.
# The 'thread_id' is used by the MemorySaver to separate and store conversation history for different users.

# --- Agent Creation with Memory ---
# Create a ReAct (Reason and Act) agent. ReAct agents are designed to reason about which tool to use and when,
# and then act by invoking the chosen tool. This allows for more complex and dynamic interactions.
agent_with_memory = create_react_agent(model, tools, checkpointer=memory) # Create a ReAct agent with the selected model, tools, and memory saver.
# - 'model': The chosen language model (OpenAI, Gemini, or Groq).
# - 'tools': The list of tools defined earlier (retriever, internet search, Wikipedia).
# - 'checkpointer=memory': The MemorySaver instance to manage conversation history.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Custom Prompt Template for Agent ---
# This prompt template defines the behavior and instructions for the ReAct agent.
# It tells the agent what tools it has, how to use them, and how to respond to user queries.
# A well-designed prompt is crucial for guiding the agent to produce desired and helpful outputs.
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
    It is crucial to avoid fabricating or speculating on information. Do not generate content beyond the retrieved data.
    Always cite your source for information when possible, e.g., mention the name of the input document that was used to retrieve the information
    or indicate if the information is from Wikipedia or the internet search.

    Here is the input from the user: {query}
    """,
    input_variables=["query"] # Input variable that the prompt template expects. This will be replaced with the user's query at runtime.
)

############§§§§§§§§§§§§§§§§§§§§§############

# --- Function to Stream Chatbot Responses ---
def stream_query_response(query, debug_mode=False, show_event_data=False, show_tool_calls=False):
    """
    Streams responses from the chatbot agent based on the user query to provide a more interactive user experience.

    Args:
        query (str): The user's input query that needs to be processed by the chatbot.
        debug_mode (bool, optional): Enables detailed debug logging to display internal agent workings. Defaults to False.
        show_event_data (bool, optional): Shows raw event data from the agent stream in an expander for advanced debugging. Defaults to False.
        show_tool_calls (bool, optional): Shows details of tool calls made by the agent in an expander to understand tool usage. Defaults to False.

    Yields:
        str: Partial responses from the chatbot, streamed as they are generated. This allows for a "typing" effect in the UI.
    """
    # --- Initialize Message History ---
    # Start with the system prompt, which sets the context and instructions for the agent.
    previous_messages = [SystemMessage(content=custom_prompt_template.format(query=query))]

    # --- Incorporate Previous Chat History ---
    # Append previous user and bot messages from session state to maintain conversation context.
    # This ensures the chatbot remembers past turns in the conversation.
    for chat in st.session_state.chat_history:
        previous_messages.append(HumanMessage(content=chat['user'])) # Add user message from history. Reconstructs user input messages.
        if chat['bot']:
            previous_messages.append(AIMessage(content=chat['bot'])) # Add bot response from history. Reconstructs previous bot responses.

    # --- Append Current User Query ---
    # Add the current user query to the message history. This is the latest user input to be processed.
    previous_messages.append(HumanMessage(content=query))

    # --- Initialize Output Accumulators ---
    full_response = "" # Accumulate the full response from the stream. Starts empty and gets built up with each streamed chunk.
    text_output = "" # Accumulate debug text output. Only used when debug_mode is enabled.
    tool_calls_output = "" # Accumulate tool calls output. Only used when show_tool_calls is enabled.

    try:
        # --- Debug Logging Setup ---
        if debug_mode:
            text_output += "Debug Log:\n"
            text_output += "--------------------\n"
            text_output += "Initial Messages to Agent:\n" # Shows the initial messages sent to the agent for debugging.
            for msg in previous_messages:
                text_output += f"- {msg}\n" # Lists each message in the initial message history.
            text_output += "\nAgent Stream Output:\n" # Marks the start of the agent's stream output in the debug log.

        # --- Stream Responses from Agent ---
        # Stream responses from the agent. 'stream' method provides incremental responses as they are generated.
        for event in agent_with_memory.stream(
            {"messages": previous_messages}, config=agent_config, stream_mode="values" # Pass messages and agent config for streaming.
            # - 'messages': The list of messages including system prompt, chat history, and current query.
            # - 'config': Agent configuration, including the unique thread ID for memory management.
            # - 'stream_mode="values"':  Configures the stream to yield only the 'value' part of events, which are typically strings or dictionaries representing messages.
        ):
            # --- Event Handling ---
            if isinstance(event, (str, dict)): # Handle string and dictionary events from the stream. LangGraph agent can emit different event types.
                if isinstance(event, dict): # Process dictionary events (typically containing structured messages, often from LangGraph).
                    if event.get('messages'): # Check if the event is a message event. LangGraph agent may emit other types of events besides messages.
                        last_message = event['messages'][-1] # Get the latest message from the event. LangGraph events can contain lists of messages.
                        full_response = last_message.content # Extract the content of the last message. This is the actual text response from the agent.

                        if debug_mode:
                            text_output += f"\n**Message Type**: {type(last_message).__name__}\n" # Log message type for debugging.
                            text_output += f"**Content**: {last_message.content}\n" # Log message content for debugging.
                            if isinstance(last_message, AIMessage) and last_message.tool_calls: # Check if it's an AI message and has tool calls. Tool calls indicate the agent is using a tool.
                                text_output += "**Tool Calls**:\n" # Indicate tool calls in debug log.
                                tool_calls_output += "**Tool Calls**:\n" # Also accumulate tool call output for separate display.
                                for tool_call in last_message.tool_calls: # Iterate through each tool call in the message.
                                    tool_call_debug_str = f"  - **Tool Name**: {tool_call['name']}\n" # Extract tool name.
                                    tool_call_debug_str += f"    **Tool Args**: {tool_call['args']}\n" # Extract tool arguments.
                                    text_output += tool_call_debug_str # Add tool call details to debug log.
                                    tool_calls_output += tool_call_debug_str # Add tool call details to separate tool calls output.
                else: # Handle string events (raw text responses, simpler stream events).
                    full_response = event # Assign the string event as the full response. This is the text content.
                    if debug_mode:
                        text_output += f"\n**String Event**: {event}\n" # Log string event in debug mode.
            elif debug_mode:
                text_output += f"\n**Event**: {event}\n" # Log other event types if in debug mode. For capturing unexpected event types.
            yield full_response # Yield the partial response to the Streamlit app. This makes the function a generator.

        # --- Post-Stream Processing ---
        # After the stream finishes, update the chat history with the full bot response.
        st.session_state.chat_history[latest_index]['bot'] = full_response # Update chat history with the full bot response. Store the complete response in the chat history.
        if debug_mode:
            st.session_state.debug_output = text_output # Store debug output in session state for display in the UI.
        if show_tool_calls:
            st.session_state.tool_calls_output = tool_calls_output # Store tool calls output in session state for display in the UI.
        if show_event_data: # Store the last event data in session state after the loop for potential raw data inspection.
            st.session_state.event_data = event # Store the last 'event' for displaying raw event data in the UI.
            # Note: Currently only storing the *last* event. Consider storing all events if needed for more detailed analysis.

    except Exception as e: # --- Error Handling ---
        logging.error(f"Error processing response: {e}", exc_info=True) # Log the error using the logging module, including traceback.
        yield "I encountered an error processing your request. Please try again later." # Yield an error message to the user in case of exceptions.
        # This provides a user-friendly error message instead of crashing the application.
    # ... (rest of stream_query_response) ...

############§§§§§§§§§§§§§§§§§§§§§############

# --- Initialize Session State Variables ---
# Streamlit's session state is used to persist variables across user interactions within a session.
# This is essential for maintaining chat history, debug outputs, and other application states
# that should persist between user queries.

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # Initialize chat history list to store user and bot messages.
    # 'chat_history' stores the conversation history as a list of dictionaries,
    # with each dictionary containing 'user' and 'bot' message pairs.

if 'debug_output' not in st.session_state:
    st.session_state.debug_output = "" # Initialize debug output string.
    # 'debug_output' stores detailed debugging logs when debug mode is enabled, for display in the UI.

if 'tool_calls_output' not in st.session_state:
    st.session_state.tool_calls_output = "" # Initialize tool calls output string.
    # 'tool_calls_output' stores details of tool interactions for display in the UI when enabled.

if 'event_data' not in st.session_state: # Initialize event_data
    st.session_state.event_data = None # Initialize event_data to None.
    # 'event_data' stores raw event data from the agent stream, primarily for advanced debugging and technical analysis.

# --- Streamlit App Title ---
st.title("LangChain Chatbot with Streamlit Frontend") # Set the title of the Streamlit application

############§§§§§§§§§§§§§§§§§§§§§############

# --- Sidebar Checkboxes and Help Section ---
with st.sidebar.expander("Help & Display Options",  expanded=True):
    show_tool_calls = st.checkbox("Show Tool Calls", value=True) # Checkbox to show tool call details
    st.caption("Display details of tools used by the chatbot to answer your query.") # Description for "Show Tool Calls"

    debug_mode = st.checkbox("Show Debug Log", value=False) # Checkbox to enable debug log display
    st.caption("Enable detailed technical logs for debugging and advanced understanding.") # Description for "Show Debug Log"

    show_event_data = st.checkbox("Show Event Data", value=False) # Checkbox to show raw event data
    st.caption("Show raw communication data from the chatbot agent (technical).") # Description for "Show Event Data"


# --- Display Chat History from Session State ---
for chat in st.session_state.chat_history: # Iterate through chat history
    with st.chat_message("user"): # Display user messages in chat format
        st.write(chat['user']) # Write user message
    if chat['bot']: # Check if there is a bot response
        with st.chat_message("assistant"): # Display bot messages in chat format
            st.write(chat['bot']) # Write bot message

# --- User Input Handling ---
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

# --- Conditional Display of Debug and Tool Call Output Expanders ---
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
