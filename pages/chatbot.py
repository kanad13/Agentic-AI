# --- Core Libraries ---
import streamlit as st  # Library for creating interactive web applications
import os  # Provides functions for interacting with the operating system
import uuid  # Module for generating universally unique identifiers
import logging # Module for logging events and errors for debugging and monitoring
from dotenv import load_dotenv  # Function to load environment variables from a .env file
import requests # Library for making HTTP requests

# --- Langchain Framework ---
from langchain_openai import ChatOpenAI  # OpenAI's chat model integration within Langchain
from langchain_google_genai import ChatGoogleGenerativeAI  # Google's Gemini chat model integration within Langchain
from langchain_groq import ChatGroq  # Groq's chat model integration within Langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Loader for PDF documents from a directory
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Tool to split text into smaller chunks recursively
from langchain_huggingface import HuggingFaceEmbeddings  # Hugging Face embeddings for converting text to vectors
from langchain_community.vectorstores import FAISS  # FAISS vector store for efficient similarity search of vectors
from langchain.tools.retriever import create_retriever_tool  # Function to create Langchain tools from retrievers
from langchain_community.tools.tavily_search import TavilySearchResults  # Tavily Search tool for web search functionality
from langchain_community.retrievers import WikipediaRetriever # Retriever to fetch content from Wikipedia
from langgraph.checkpoint.memory import MemorySaver  # For persisting agent states in memory, enabling conversational history
from langgraph.prebuilt import create_react_agent  # Function to create ReAct agents using Langchain
from langchain_core.prompts import PromptTemplate  # Class for creating and managing prompt templates for LLMs
from langchain.schema import AIMessage, HumanMessage, SystemMessage # Message types for structuring conversations with language models

# --- Overall Comment ---
# Import necessary libraries for chatbot functionality, including Streamlit for UI,
# Langchain for LLM interactions, vector database management, and utility libraries.
# These libraries facilitate building a chatbot with document retrieval, internet search, and conversational memory.


# --- Environment Variables ---
# Load environment variables from .env file.
# This is crucial for securely managing API keys and other sensitive configurations outside of the main codebase, improving security and deployment flexibility.
load_dotenv()

# --- Basic Error Logging ---
logging.basicConfig(level=logging.ERROR) # Configure basic logging to capture errors.
# Setting the logging level to ERROR ensures that only error messages and above (critical, fatal) are captured. This is useful for debugging and monitoring the application for critical issues.
# Errors will be logged to the console or wherever logging is configured to output.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="ChatBot", # Sets the title displayed in the browser tab
    page_icon=":chat-plus-outline:", # Sets the icon displayed in the browser tab
    layout="wide", # Configures the page layout to use the full width of the screen
    initial_sidebar_state="expanded", # Sets the sidebar to be expanded by default when the app loads
    menu_items=None # Hides the default Streamlit menu in the top right corner
)

# --- API Key Retrieval ---
# Retrieve API keys from environment variables.
# These keys are essential for authenticating with various AI models and services.
# Ensure these API keys are correctly set in your .env file.
# Refer to the respective provider's documentation for obtaining these keys.
# API keys are loaded using `os.getenv()` and are necessary for accessing services like OpenAI, Google Gemini, Groq, Tavily Search, and GitHub.

GROQ_API_KEY = os.getenv('GROQ_API_KEY') # API key for Groq models. Required for accessing Groq's language models.
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') # API key for Google models. Required for accessing Google's Gemini language models.
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # API key for OpenAI models. Required for accessing OpenAI's language models like GPT-4o.
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY') # API key for Tavily Search. Required for using the Tavily internet search tool.
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY') # API key for Langchain Observability/Tracing features. Enables advanced debugging and monitoring of Langchain applications.
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2') # Flag to enable Langchain Tracing V2 for more detailed observability. Set to 'true' to activate. Useful for debugging and performance analysis.
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN_kanad1323') # API token for GitHub issue creation. Required for creating issues in the specified GitHub repository.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Model Selection in Sidebar ---
st.sidebar.title("Settings") # Sets the title for the sidebar section
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=["gpt-4o-mini", "gemini-2.0-flash-exp", "gemma2-9b-it"] # Dropdown to choose the chatbot model.
    # Model options are selected for a balance of performance and cost.
    # 'gpt-4o-mini' represents OpenAI's latest model, 'gemini-2.0-flash-exp' is Google's fast model,
    # and 'gemma2-9b-it' is a powerful open-source model from Groq.
    # Users can select their preferred LLM from this dropdown in the sidebar.
)

# Initialize Chat Model based on user selection
if selected_model == "gpt-4o-mini":
    model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY) # Use OpenAI's GPT-4o model. Integrates with OpenAI's API using the selected model.
elif selected_model == "gemini-2.0-flash-exp":
    model = ChatGoogleGenerativeAI(model ="gemini-2.0-flash-exp", google_api_key=GOOGLE_API_KEY) # Use Google's Gemini model. Integrates with Google's Gemini API.
elif selected_model == "gemma2-9b-it":
    model = ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY) # Use Groq's Gemma model. Integrates with Groq's API.
# Based on the `selected_model`, the appropriate Langchain chat model is initialized with its respective API key.

# --- Performance Tuning ---
# Configure Tokenizers Parallelism and Embedding Batch Size
os.environ["TOKENIZERS_PARALLELISM"] = "true" # Enable parallel tokenization for potentially faster processing.
# This can speed up text processing, especially when dealing with large documents, by utilizing multiple CPU cores.
# However, it might slightly increase resource consumption.
# Setting TOKENIZERS_PARALLELISM to "true" can improve performance for text processing tasks, especially with large inputs.

embedding_batch_size = 512 # Set batch size for embedding operations. Adjust based on available memory and GPU (if applicable).
# Larger batch sizes can improve embedding generation speed, but require more memory.
# A value of 512 is a reasonable starting point, but you may need to decrease it if you encounter memory issues,
# especially on systems with limited RAM or when processing very large documents.
# `embedding_batch_size` controls how many text chunks are processed in parallel during embedding generation.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Document Loading ---
# Document Loading Function
@st.cache_resource # Cache the output of this function to avoid reloading documents on every run.
# This decorator from Streamlit efficiently caches the result of 'load_documents'.
# Subsequent runs of the app will reuse the loaded documents unless the function's inputs change.
# Caching `load_documents` significantly improves app performance by avoiding redundant document loading.
def load_documents():
    """Loads PDF documents from the './input_files/' directory."""
    loader = PyPDFDirectoryLoader(path="./input_files/") # Load PDF documents from the specified directory using Langchain's PyPDFDirectoryLoader.
    return loader.load() # Return the list of loaded documents.
# `load_documents` function uses PyPDFDirectoryLoader to load all PDF files from the './input_files/' directory.

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
    # `add_start_index=True` is helpful for referencing the original document location of retrieved information.
)

document_chunks = text_splitter.split_documents(loaded_documents) # Split loaded documents into smaller, manageable chunks.
# `RecursiveCharacterTextSplitter` breaks down the loaded documents into smaller chunks, which are better suited for embedding and retrieval.

# --- Vector Store Creation ---
# Vector Store Creation Function
@st.cache_resource # Cache the output of this function to avoid recreating the vector store on every run.
# Caching is crucial for performance as vector store creation can be computationally expensive.
# Caching `create_vectorstore` is important because vector store creation is a resource-intensive operation.
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
# `create_vectorstore` function generates embeddings for the document chunks and stores them in a FAISS vector store for efficient retrieval.

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
    # `k=6` determines the number of document chunks to retrieve during similarity search.
)
# `vectorstore.as_retriever()` creates a retriever object configured for similarity search, which will be used to fetch relevant documents.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Tool Definitions for Agent ---
# Tools enhance the agent's capabilities, allowing it to interact with the outside world
# and access information beyond its internal knowledge base.
# This is crucial for creating a more versatile and informative chatbot.
# Defining tools expands the agent's functionality beyond just language generation, enabling it to perform specific tasks.

# Document Retrieval Tool
# This tool allows the agent to retrieve information from the loaded documents (PDF files).
retriever_tool = create_retriever_tool(
    document_retriever, # The retriever object created earlier, responsible for fetching document chunks.
    "retriever_tool", # Name of the tool, used by the agent to invoke it. This name is referenced in the agent's prompt.
    "Retrieves and provides information from the input documents." # Description of the tool, used by the agent to understand its purpose and when to use it.
)
# `retriever_tool` enables the agent to access and use the document retriever to find information within the loaded documents.

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
# `internet_search_tool` uses Tavily Search API to equip the agent with real-time web search capabilities.

# Wikipedia Retrieval Tool
# This tool allows the agent to retrieve information from Wikipedia articles.
wikipedia_retriever = WikipediaRetriever() # Initialize Wikipedia retriever using Langchain's built-in Wikipedia retriever.
wikipedia_retriever_tool = create_retriever_tool(
    wikipedia_retriever, # The Wikipedia retriever object.
    "wikipedia_retriever_tool", # Name of the Wikipedia retrieval tool.
    "Retrieves and provides information from Wikipedia articles." # Description of the tool.
)
# `wikipedia_retriever_tool` provides the agent with access to information from Wikipedia using Langchain's WikipediaRetriever.

# List of Tools for the Agent
# Combine all defined tools into a list. This list will be passed to the agent creation function.
tools = [retriever_tool, wikipedia_retriever_tool, internet_search_tool] # List of tools available to the agent.
# The agent will decide which tool to use based on the user's query and the tool descriptions.
# The order in this list doesn't typically matter for functionality but can be organized logically.
# `tools` list aggregates all the tools that the agent can utilize to answer user queries.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Memory Setup for Conversation History ---
# Memory is crucial for creating conversational chatbots that can remember past interactions.
# MemorySaver is used here to persist conversation history, allowing the chatbot to maintain context across multiple turns.
memory = MemorySaver() # Initialize MemorySaver to persist conversation history using Langchain's MemorySaver.
# `MemorySaver` is initialized to enable the agent to remember and refer back to previous interactions in the conversation.

# --- Unique Thread ID for Conversation Management ---
# A unique thread ID is generated for each user session to isolate conversations and manage state independently.
if 'unique_id' not in st.session_state:
    st.session_state.unique_id = str(uuid.uuid4()) # Generate a unique ID if not already present in session state using UUID.
agent_config = {"configurable": {"thread_id": st.session_state.unique_id}} # Configuration for the agent, including the unique thread ID.
# The 'thread_id' is used by the MemorySaver to separate and store conversation history for different users.
# A unique `thread_id` ensures that each user's conversation is isolated and managed separately in the memory.

# --- Agent Creation with Memory ---
# Create a ReAct (Reason and Act) agent. ReAct agents are designed to reason about which tool to use and when,
# and then act by invoking the chosen tool. This allows for more complex and dynamic interactions.
agent_with_memory = create_react_agent(model, tools, checkpointer=memory) # Create a ReAct agent with the selected model, tools, and memory saver.
# - 'model': The chosen language model (OpenAI, Gemini, or Groq).
# - 'tools': The list of tools defined earlier (retriever, internet search, Wikipedia).
# - 'checkpointer=memory': The MemorySaver instance to manage conversation history.
# `create_react_agent` sets up the agent with the selected LLM, tools, and memory, enabling it to reason and act during conversations.

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
    # The prompt template instructs the agent on its role, available tools, and the process for answering user queries.
)
# `custom_prompt_template` defines the core instructions and behavior of the agent, including tool usage and response strategies.

############§§§§§§§§§§§§§§§§§§§§§############

# --- GitHub Issue Creation Function ---
def create_github_issue(title, body, tool_calls_comment=None, debug_log_comment=None, event_data_comment=None):
    """
    Creates a GitHub issue in the specified repository with the given title and body.
    Optionally adds comments for tool calls, debug logs, and event data.

    Args:
        title (str): The title of the GitHub issue (user's question).
        body (str): The body of the GitHub issue (chatbot's answer).
        tool_calls_comment (str, optional): Comment text for tool calls. Defaults to None.
        debug_log_comment (str, optional): Comment text for debug log. Defaults to None.
        event_data_comment (str, optional): Comment text for event data. Defaults to None.

    Returns:
        bool: True if issue creation and comment addition were successful, False otherwise.
    """
    repo_owner = "kanad1323"  # Your GitHub username/organization. Replace with your actual username or organization.
    repo_name = "test-repo"   # Your repository name. Replace with your actual repository name.
    github_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues" # GitHub API endpoint for creating issues.
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}', # Include GitHub token for authentication.
        'Accept': 'application/vnd.github.v3+json' # Specify API version and response format.
    }
    issue_data = {
        'title': title, # Issue title is set to the user's query.
        'body': body # Issue body is set to the chatbot's response.
    }

    try:
        response = requests.post(github_api_url, headers=headers, json=issue_data) # Send POST request to create the issue.
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx). This will handle API errors.
        issue_json = response.json() # Parse the JSON response from GitHub API.
        issue_number = issue_json.get('number') # Extract the issue number from the JSON response.

        if issue_number:
            logging.info(f"GitHub issue created successfully: {issue_json.get('html_url')}") # Log successful issue creation with URL.

            comments_to_add = {
                "Show Tool Calls": tool_calls_comment, # Comment for tool call details, if available.
                "Show Debug Log": debug_log_comment, # Comment for debug log, if available.
                "Show Event Data": event_data_comment, # Comment for event data, if available.
            }

            for comment_title, comment_text in comments_to_add.items(): # Iterate through comments to add.
                if comment_text: # Only add comment if text is not None.
                    comment_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/comments" # API endpoint for issue comments.
                    comment_data = {'body': comment_text} # Comment body is set to the provided comment text.
                    comment_response = requests.post(comment_api_url, headers=headers, json=comment_data) # Send POST request to add comment.
                    if not comment_response.ok: # Check if comment creation was successful.
                        logging.error(f"Error adding comment '{comment_title}' to GitHub issue: {comment_response.status_code} - {comment_response.text}") # Log comment creation error.
                    else:
                        logging.info(f"Comment '{comment_title}' added to GitHub issue.") # Log successful comment creation.
            return True # Issue and comments created successfully.

        else:
            logging.error(f"Could not retrieve issue number from GitHub API response.") # Log error if issue number is not found.
            return False # Issue creation likely failed, but no issue number to add comments.

    except requests.exceptions.RequestException as e: # Catch exceptions related to HTTP requests.
        logging.error(f"Error creating GitHub issue: {e}") # Log request exceptions during issue creation.
        return False # Issue creation failed due to request exception.

    except Exception as e: # Catch any other unexpected exceptions.
        logging.error(f"An unexpected error occurred during GitHub issue creation: {e}") # Log unexpected errors during issue creation.
        return False # Unexpected error.


############§§§§§§§§§§§§§§§§§§§§§############

# --- Function to Stream Chatbot Responses ---
def stream_query_response(query, debug_mode=False, show_event_data=False, show_tool_calls=False):
    """
    Streams responses from the chatbot agent based on the user query to provide a more interactive user experience.
    Also creates a GitHub issue with the question and answer, including debug/tool call information as comments if enabled.

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
    # The initial `previous_messages` list starts with the `SystemMessage` containing the prompt template, setting the agent's instructions.

    # --- Incorporate Previous Chat History ---
    # Append previous user and bot messages from session state to maintain conversation context.
    # This ensures the chatbot remembers past turns in the conversation.
    for chat in st.session_state.chat_history:
        previous_messages.append(HumanMessage(content=chat['user'])) # Add user message from history. Reconstructs user input messages.
        if chat['bot']:
            previous_messages.append(AIMessage(content=chat['bot'])) # Add bot response from history. Reconstructs previous bot responses.
    # Previous conversation turns from `st.session_state.chat_history` are added to `previous_messages` to maintain context.

    # --- Append Current User Query ---
    # Add the current user query to the message history. This is the latest user input to be processed.
    previous_messages.append(HumanMessage(content=query))
    # The current user's query is appended to the `previous_messages` list as a `HumanMessage`.

    # --- Initialize Output Accumulators ---
    full_response = "" # Accumulate the full response from the stream. Starts empty and gets built up with each streamed chunk.
    text_output = "" # Accumulate debug text output. Only used when debug_mode is enabled.
    tool_calls_output = "" # Accumulate tool calls output. Only used when show_tool_calls is enabled.
    # Output accumulators are initialized to store the streamed response, debug logs, and tool call details.

    try:
        # --- Debug Logging Setup ---
        if debug_mode:
            text_output += "Debug Log:\n"
            text_output += "--------------------\n"
            text_output += "Initial Messages to Agent:\n" # Shows the initial messages sent to the agent for debugging.
            for msg in previous_messages:
                text_output += f"- {msg}\n" # Lists each message in the initial message history.
            text_output += "\nAgent Stream Output:\n" # Marks the start of the agent's stream output in the debug log.
        # Debug logging is set up to capture initial messages and agent stream output when `debug_mode` is enabled.

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
        # The `agent_with_memory.stream` method is used to get streamed responses from the agent.
        # The loop iterates through events, handling different event types (string, dictionary) and extracting the response content.
        # Debug and tool call information are captured if respective flags are enabled.
        # Partial responses are yielded to update the UI in real-time.

        # --- GitHub Issue Creation ---
        tool_calls_comment_text = f"**Show Tool Calls:**\n```\n{st.session_state.tool_calls_output}\n```" if show_tool_calls and st.session_state.tool_calls_output else None # Format tool calls output for comment.
        debug_log_comment_text = f"**Show Debug Log:**\n```\n{st.session_state.debug_output}\n```" if debug_mode and st.session_state.debug_output else None # Format debug log output for comment.
        event_data_comment_text = f"**Show Event Data:**\n```\n{st.session_state.event_data}\n```" if show_event_data and st.session_state.event_data else None # Format event data for comment.

        issue_created = create_github_issue(
            title=query,  # User's question as issue title.
            body=full_response, # Bot's answer as issue body.
            tool_calls_comment=tool_calls_comment_text, # Tool calls information as comment.
            debug_log_comment=debug_log_comment_text, # Debug log information as comment.
            event_data_comment=event_data_comment_text # Event data as comment.
        ) # Call the function to create GitHub issue with collected information.

        if issue_created:
            logging.info("GitHub issue creation process completed.") # Log successful GitHub issue creation.
        else:
            logging.error("GitHub issue creation process failed.") # Log failure of GitHub issue creation.


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
        # After the streaming is complete, the full response is stored in `st.session_state.chat_history`.
        # Debug output, tool call output, and event data are also stored in session state if enabled for display in the UI.

    except Exception as e: # --- Error Handling ---
        logging.error(f"Error processing response: {e}", exc_info=True) # Log the error using the logging module, including traceback.
        yield "I encountered an error processing your request. Please try again later." # Yield an error message to the user in case of exceptions.
        # This provides a user-friendly error message instead of crashing the application.
        # Error handling is implemented to catch exceptions during response processing, log the error, and yield a user-friendly error message.
    # ... (rest of stream_query_response) ...
# `stream_query_response` function handles sending the user query to the agent, streaming back the response, updating chat history, and handling errors.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Initialize Session State Variables ---
# Streamlit's session state is used to persist variables across user interactions within a session.
# This is essential for maintaining chat history, debug outputs, and other application states
# that should persist between user queries.
# Session state initialization ensures that necessary variables are set up when the app starts, allowing for state persistence.

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # Initialize chat history list to store user and bot messages.
    # 'chat_history' stores the conversation history as a list of dictionaries,
    # with each dictionary containing 'user' and 'bot' message pairs.
    # `chat_history` in session state stores the ongoing conversation as a list of user and bot message pairs.

if 'debug_output' not in st.session_state:
    st.session_state.debug_output = "" # Initialize debug output string.
    # 'debug_output' stores detailed debugging logs when debug mode is enabled, for display in the UI.
    # `debug_output` stores debug log messages for display when debug mode is active.

if 'tool_calls_output' not in st.session_state:
    st.session_state.tool_calls_output = "" # Initialize tool calls output string.
    # 'tool_calls_output' stores details of tool interactions for display in the UI when enabled.
    # `tool_calls_output` stores tool interaction details for display when tool calls are shown in the UI.

if 'event_data' not in st.session_state: # Initialize event_data
    st.session_state.event_data = None # Initialize event_data to None.
    # 'event_data' stores raw event data from the agent stream, primarily for advanced debugging and technical analysis.
    # `event_data` stores raw communication events from the agent, intended for detailed debugging.

# --- Streamlit App Title ---
# st.title("Understand AI Agents") # Set the title of the Streamlit application
st.write("Need some text on what this page shows") # Placeholder for app description, consider replacing with a relevant description.
# `st.write` with "Need some text on what this page shows" is a placeholder and should be replaced with a descriptive text about the app.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Sidebar Checkboxes and Help Section ---
with st.sidebar.expander("Help & Display Options",  expanded=True):
    show_tool_calls = st.checkbox("Show Tool Calls", value=True) # Checkbox to show tool call details
    st.caption("Display details of tools used by the chatbot to answer your query.") # Description for "Show Tool Calls"
    # `show_tool_calls` checkbox in the sidebar allows users to toggle the display of tool interaction details.

    debug_mode = st.checkbox("Show Debug Log", value=True) # Checkbox to enable debug log display
    st.caption("Enable detailed technical logs for debugging and advanced understanding.") # Description for "Show Debug Log"
    # `debug_mode` checkbox enables or disables the display of detailed debug logs.

    show_event_data = st.checkbox("Show Event Data", value=True) # Checkbox to show raw event data
    st.caption("Show raw communication data from the chatbot agent (technical).") # Description for "Show Event Data"
    # `show_event_data` checkbox controls the display of raw agent communication events.

# --- Display Chat History from Session State ---
for chat in st.session_state.chat_history: # Iterate through chat history
    with st.chat_message("user"): # Display user messages in chat format
        st.write(chat['user']) # Write user message
    if chat['bot']: # Check if there is a bot response
        with st.chat_message("assistant"): # Display bot messages in chat format
            st.write(chat['bot']) # Write bot message
# This loop iterates through `st.session_state.chat_history` and displays each message turn in the chat UI.

# --- User Input Handling ---
if user_input := st.chat_input("You:"):
    # Append user message to chat history
    st.session_state.chat_history.append({"user": user_input, "bot": ""})
    latest_index = len(st.session_state.chat_history) - 1
    # When user provides input via `st.chat_input`, the input is appended to `st.session_state.chat_history`.

    # Display User Message in Chat
    with st.chat_message("user"):
        st.write(user_input)
    # The user's input is immediately displayed in the chat UI.

    # Placeholder for Bot Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
          for response in stream_query_response(user_input, debug_mode=debug_mode, show_event_data=show_event_data, show_tool_calls=show_tool_calls):
            full_response = response
            response_placeholder.markdown(full_response)
    # A placeholder is created for the bot's response, and `stream_query_response` is called to get streamed responses, which are then displayed in the placeholder.
else: # Add this 'else' block to handle the case when there's no user input from chat_input
    st.write("Start by asking these sample questions or ask your own question:") # Optional: Add a line to guide users
    # If no user input is provided via `st.chat_input`, this message guides users to ask questions or use sample questions.

    col1, col2, col3 = st.columns(3) # Create 3 columns for buttons
    # Three columns are created to arrange sample question buttons.

    with col1:
        if st.button("Can I see aeroplanes flying in Berlin sky right now?", key="sample_1"): # Unique key for each button
            user_input = "Can I see aeroplanes flying in Berlin sky right now taking into consideration the current weather in Berlin?"
    # Sample question button in the first column, setting `user_input` when clicked.

    with col2:
        if st.button("What is Model Collapse?", key="sample_2"): # Unique key for each button
            user_input = "Breifly explain concept of model collapse."
    # Sample question button in the second column.

    with col3:
        if st.button("Is Laptop Man stronger than Superman?", key="sample_3"): # Unique key for each button
            user_input = "Who is Laptop Man? Where did you find information about him? Who is Superman? Where did you find information about him? Is Laptop Man stronger than Superman?"
    # Sample question button in the third column.

    if 'user_input' not in locals(): # If no sample question button was clicked, initialize user_input to None.
        user_input = None # Ensure user_input is None if no button is pressed and no chat input is given.
    # If no sample question button is clicked and no input from `st.chat_input`, `user_input` is set to `None`.

    if user_input: # Now the rest of your input processing logic will only run if there's a user_input (either from chat or sample button)
        # Append user message to chat history
        st.session_state.chat_history.append({"user": user_input, "bot": ""})
        latest_index = len(st.session_state.chat_history) - 1
        # If `user_input` is set (either from sample button or chat input), it's appended to `st.session_state.chat_history`.

        # Display User Message in Chat
        with st.chat_message("user"):
            st.write(user_input)
        # User input is displayed in the chat UI.

        # Placeholder for Bot Response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
              for response in stream_query_response(user_input, debug_mode=debug_mode, show_event_data=show_event_data, show_tool_calls=show_tool_calls):
                full_response = response
                response_placeholder.markdown(full_response)
        # Bot response is streamed and displayed in the chat UI using `stream_query_response`.

# --- Conditional Display of Debug and Tool Call Output Expanders ---
if show_tool_calls: # Conditionally display tool calls output
  if st.session_state.tool_calls_output: # Check if there is tool calls output to display
    with st.expander("Tool Interaction Details"):
        st.write("This section reveals the tools the chatbot used to respond to your query. It shows which tools were activated and what instructions were given to them. This can help you understand how the chatbot is working behind the scenes to find information.")
        st.code(st.session_state.tool_calls_output) # Display tool calls output in an expander
# If `show_tool_calls` is enabled and there's `tool_calls_output`, it is displayed in an expander.

if debug_mode: # Conditionally display debug output
  if st.session_state.debug_output: # Check if there is debug output to display
    with st.expander("Detailed Debugging Information"):
        st.write("This section provides a detailed technical log of the chatbot's thought process. It's useful for understanding exactly what steps the chatbot took to answer your question, including the messages sent back and forth internally. This level of detail is generally for debugging and advanced understanding.")
        st.code(st.session_state.debug_output) # Display debug output in an expander
# If `debug_mode` is enabled and there's `debug_output`, it is displayed in an expander.

if show_event_data: # Conditionally display event data expander
  if st.session_state.event_data: # Check if there is event data to display
    with st.expander("Raw Agent Communication Data (Technical)"):
        st.write("This section displays the raw, technical data stream from the chatbot agent. This is advanced debugging information showing the step-by-step communication within the agent as it processes your request. It's primarily useful for developers or those deeply interested in the technical workings.")
        st.write("Event Details:", st.session_state.event_data) # Display stored event data from session state
# If `show_event_data` is enabled and there's `event_data`, it is displayed in an expander.
