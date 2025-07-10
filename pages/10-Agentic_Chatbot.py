# --- Core Libraries ---
import streamlit as st  # For interactive web applications
import os  # Operating system interactions
import uuid  # Generate unique identifiers
import logging # Logging events and errors
from dotenv import load_dotenv  # Load environment variables from .env file
import requests # For making HTTP requests
import time # For adding delays

# --- Langchain Framework ---
from langchain_openai import ChatOpenAI  # OpenAI chat model integration
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini chat model integration
from langchain_groq import ChatGroq  # Groq chat model integration
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Load PDF documents from directory
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split text into smaller chunks recursively
from langchain_huggingface import HuggingFaceEmbeddings  # Hugging Face embeddings for text to vectors
from langchain_community.vectorstores import FAISS  # FAISS vector store for efficient similarity search
from langchain.tools.retriever import create_retriever_tool  # Create Langchain tools from retrievers
from langchain_tavily import TavilySearch  # Tavily Search tool for web search
from langchain_community.retrievers import WikipediaRetriever # Retriever for Wikipedia content
from langgraph.checkpoint.memory import MemorySaver  # Persist agent states in memory for conversation history
from langgraph.prebuilt import create_react_agent  # Create ReAct agents
from langchain_core.prompts import PromptTemplate  # Create and manage prompt templates
from langchain.schema import AIMessage, HumanMessage, SystemMessage # Message types for LLM conversations


# --- Environment Variables ---
load_dotenv() # Load environment variables from .env file. Securely manage API keys and configurations.

# --- Basic Error Logging ---
logging.basicConfig(level=logging.ERROR) # Configure basic logging for errors and above. Useful for monitoring critical issues.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="ChatBot", # Browser tab title
    page_icon=":chat-plus-outline:", # Browser tab icon
    layout="wide", # Full width layout
    initial_sidebar_state="expanded", # Expanded sidebar by default
    menu_items=None # Hide default Streamlit menu
)

# --- API Key Retrieval ---
# Retrieve API keys from environment variables. Essential for authenticating with AI models and services.
# Ensure API keys are set in your .env file as per provider documentation.
GROQ_API_KEY = os.getenv('GROQ_API_KEY') # Groq API key.
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') # Google API key.
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # OpenAI API key.
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY') # Tavily Search API key.
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY') # Langchain Observability/Tracing API key.
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2') # Langchain Tracing V2 flag ('true' to enable).
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN_kanad1323') # GitHub API token for issue creation.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Model Selection in Sidebar ---
st.sidebar.title("Settings") # Sidebar title
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=["OpenAI GPT-4o", "Meta Llama-3.1", "Google Gemma-2"] # Model options: OpenAI, Google, Groq.
)

# Initialize Chat Model based on user selection
if selected_model == "OpenAI GPT-4o":
    model = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", openai_api_key=OPENAI_API_KEY) # Use OpenAI GPT-4o model.
elif selected_model == "Meta Llama-3.1":
    model = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY) # Use Groq Gemma model.
elif selected_model == "Google Gemma-2":
    model = ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY) # Use Groq Gemma model.

# --- Performance Tuning ---
# Configure Tokenizers Parallelism and Embedding Batch Size
os.environ["TOKENIZERS_PARALLELISM"] = "true" # Enable parallel tokenization for potential speedup in text processing. May slightly increase resource usage.
embedding_batch_size = 512 # Batch size for embedding operations. Adjust based on memory/GPU. Larger batches can be faster but require more memory. Start with 512 and reduce if memory issues occur.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Document Loading ---
# Document Loading Function
@st.cache_resource # Cache output to avoid reloading documents on every run.
def load_documents():
    """Loads PDF documents from the './input_files/' directory."""
    loader = PyPDFDirectoryLoader(path="./input_files/") # Load PDFs from directory.
    return loader.load() # Return loaded documents.

loaded_documents = load_documents() # Load documents (cached).

# --- Document Chunking ---
# Document Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Maximum chunk size (characters). Common starting point: 1000. Balances context and processing.
    chunk_overlap=200, # Overlap between chunks (characters) to maintain context continuity. Helps prevent loss of context at chunk boundaries.
    add_start_index=True # Include start index in original document for each chunk. Useful for source tracing and citations.
)

document_chunks = text_splitter.split_documents(loaded_documents) # Split documents into chunks.

# --- Vector Store Creation ---
# Vector Store Creation Function
@st.cache_resource # Cache output to avoid recreating vector store on every run.
def create_vectorstore():
    """Creates and caches a FAISS vector store from document chunks using HuggingFace embeddings."""
    # --- Embedding Model ---
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2", # Pre-trained sentence embedding model from Hugging Face.
        encode_kwargs={'batch_size': embedding_batch_size} # Batch size for embedding generation.
    )
    # --- FAISS Vector Store ---
    return FAISS.from_documents(documents=document_chunks, embedding=embedding_model) # Create FAISS vector store.

vectorstore = create_vectorstore() # Create and load vector store (cached).

# --- Document Retriever ---
# Document Retriever
document_retriever = vectorstore.as_retriever(
    search_type="similarity", # Use similarity search to find semantically similar document chunks.
    search_kwargs={"k": 6} # Retrieve top 6 most similar document chunks (k=6). Adjust 'k' for precision/recall trade-off.
)

############§§§§§§§§§§§§§§§§§§§§§############

# --- Tool Definitions for Agent ---
# Tools for agent to interact with external world and access information.
# Expands agent functionality beyond language generation.

# Document Retrieval Tool
retriever_tool = create_retriever_tool(
    document_retriever, # Retriever object.
    "retriever_tool", # Tool name for agent.
    "Retrieves information from the input documents." # Tool description for agent.
)

# Internet Search Tool (Tavily)
internet_search_tool = TavilySearch(
    max_results=2, # Limit search results to 2.
    search_depth="advanced", # More thorough internet search.
    include_answer=True, # Include direct answer from search results if available.
    include_raw_content=True, # Include raw content of search results.
    include_images=True, # Allow image results (potential future feature).
)

# Wikipedia Retrieval Tool
wikipedia_retriever = WikipediaRetriever() # Wikipedia retriever.
wikipedia_retriever_tool = create_retriever_tool(
    wikipedia_retriever, # Wikipedia retriever object.
    "wikipedia_retriever_tool", # Tool name.
    "Retrieves information from Wikipedia articles." # Tool description.
)

# List of Tools for Agent: Tools available to the agent (retriever, Wikipedia, internet search). Agent decides tool usage based on query and tool descriptions.
tools = [retriever_tool, wikipedia_retriever_tool, internet_search_tool]

############§§§§§§§§§§§§§§§§§§§§§############

# --- Memory Setup for Conversation History ---
memory = MemorySaver() # Initialize MemorySaver for conversation history persistence.

# --- Unique Thread ID for Conversation Management ---
if 'unique_id' not in st.session_state:
    st.session_state.unique_id = str(uuid.uuid4()) # Generate unique ID if not in session state.
agent_config = {"configurable": {"thread_id": st.session_state.unique_id}} # Agent config with unique thread ID for memory.

# --- Agent Creation with Memory ---
agent_with_memory = create_react_agent(model, tools, checkpointer=memory) # Create ReAct agent with model, tools, and memory.

############§§§§§§§§§§§§§§§§§§§§§############

# --- Custom Prompt Template for Agent ---
custom_prompt_template = PromptTemplate(
    template="""
    You are an AI assistant equipped with tools: retriever_tool, wikipedia_retriever_tool, internet_search_tool.

    Tool Descriptions:
    - retriever_tool: Retrieves information from input documents (RAG).
    - wikipedia_retriever_tool: Retrieves information from Wikipedia articles.
    - internet_search_tool: Conducts real-time internet searches using Tavily Search.

    Instructions for Tool Usage:
    1. Document Retrieval: First, use `retriever_tool` to check input documents for the answer.
    2. Wikipedia Search: If no answer from documents, use `wikipedia_retriever_tool` to search Wikipedia.
    3. Internet Search: If no answer from documents or Wikipedia, use `internet_search_tool` for broader internet search.

    Fallback Response:
    If no answer found after checking all sources, respond: "I'm sorry, but I don't have the information to answer that question."

    Important Constraints:
    Avoid fabrication or speculation. Do not generate content beyond retrieved data.
    Cite sources when possible (document name, Wikipedia, internet search).

    User Input: {query}
    """,
    input_variables=["query"] # Input variable for user query.
)

############§§§§§§§§§§§§§§§§§§§§§############

# --- GitHub Issue Creation Function ---
def create_github_issue(title, body, tool_calls_comment=None, debug_log_comment=None, event_data_comment=None):
    """
    Creates a GitHub issue with title and body. Optionally adds comments for tool calls, debug logs, and event data.
    """
    repo_owner = "kanad1323"  # Your GitHub username/organization.
    repo_name = "test-repo"   # Your repository name.
    github_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues" # GitHub API endpoint.
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}', # GitHub token.
        'Accept': 'application/vnd.github.v3+json' # GitHub API v3, JSON format.
    }
    issue_data = {
        'title': title, # Issue title is user query.
        'body': body # Issue body is chatbot response.
    }

    try:
        response = requests.post(github_api_url, headers=headers, json=issue_data) # Create issue via POST.
        response.raise_for_status()  # Raise HTTPError for bad responses.
        issue_json = response.json() # Parse JSON response.
        issue_number = issue_json.get('number') # Extract issue number.

        if issue_number:
            logging.info(f"GitHub issue created: {issue_json.get('html_url')}") # Log issue creation.
            
            # Add delay to allow GitHub API to propagate the issue
            time.sleep(2)

            comments_to_add = {
                "Show Tool Calls": tool_calls_comment, # Tool calls comment.
                "Show Debug Log": debug_log_comment, # Debug log comment.
                "Show Event Data": event_data_comment, # Event data comment.
            }

            for comment_title, comment_text in comments_to_add.items(): # Add comments.
                if comment_text:
                    comment_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/comments" # Issue comments API endpoint.
                    comment_data = {'body': comment_text} # Comment body.
                    comment_response = requests.post(comment_api_url, headers=headers, json=comment_data) # Create comment via POST.
                    if not comment_response.ok:
                        logging.error(f"Error adding comment '{comment_title}': {comment_response.status_code} - {comment_response.text}") # Log comment error.
                    else:
                        logging.info(f"Comment '{comment_title}' added.") # Log comment success.
            return True # Issue and comments created successfully.

        else:
            logging.error(f"Could not get issue number from GitHub API.") # Log issue number error.
            return False # Issue creation failed (no issue number).

    except requests.exceptions.RequestException as e: # Handle request exceptions.
        logging.error(f"Error creating GitHub issue: {e}") # Log request error.
        return False # Issue creation failed due to request exception.

    except Exception as e: # Handle unexpected exceptions.
        logging.error(f"Unexpected error during GitHub issue creation: {e}") # Log unexpected error.
        return False # Issue creation failed due to unexpected error.


############§§§§§§§§§§§§§§§§§§§§§############

# --- Function to Stream Chatbot Responses ---
def stream_query_response(query, debug_mode=False, show_event_data=False, show_tool_calls=False):
    """
    Streams chatbot responses, optionally creating a GitHub issue with debug/tool call info.
    """
    # --- Initialize Message History ---
    previous_messages = [SystemMessage(content=custom_prompt_template.format(query=query))] # Initial message: system prompt.

    # --- Incorporate Previous Chat History ---
    for chat in st.session_state.chat_history:
        previous_messages.append(HumanMessage(content=chat['user'])) # User message from history.
        if chat['bot']:
            previous_messages.append(AIMessage(content=chat['bot'])) # Bot response from history.

    # --- Append Current User Query ---
    previous_messages.append(HumanMessage(content=query)) # Current user query.

    # --- Initialize Output Accumulators ---
    full_response = "" # Accumulate full response.
    text_output = "" # Accumulate debug text output.
    tool_calls_output = "" # Accumulate tool calls output.

    try:
        # --- Debug Logging Setup ---
        if debug_mode:
            text_output += "Debug Log:\n--------------------\n"
            text_output += "Initial Messages to Agent:\n" # Initial agent messages.
            for msg in previous_messages:
                text_output += f"- {msg}\n"
            text_output += "\nAgent Stream Output:\n" # Agent stream output start.

        # --- Stream Responses from Agent ---
        for event in agent_with_memory.stream(
            {"messages": previous_messages}, config=agent_config, stream_mode="values" # Stream agent responses.
        ):
            # --- Event Handling ---
            if isinstance(event, (str, dict)):
                if isinstance(event, dict):
                    if event.get('messages'):
                        last_message = event['messages'][-1] # Get latest message.
                        full_response = last_message.content # Extract response content.

                        if debug_mode:
                            text_output += f"\n**Message Type**: {type(last_message).__name__}\n" # Log message type.
                            text_output += f"**Content**: {last_message.content}\n" # Log message content.
                            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                                text_output += "**Tool Calls**:\n" # Tool calls start.
                                tool_calls_output += "**Tool Calls**:\n"
                                for tool_call in last_message.tool_calls:
                                    tool_call_debug_str = f"  - **Tool Name**: {tool_call['name']}\n" # Tool name.
                                    tool_call_debug_str += f"    **Tool Args**: {tool_call['args']}\n" # Tool args.
                                    text_output += tool_call_debug_str
                                    tool_calls_output += tool_call_debug_str
                else:
                    full_response = event # String event is full response.
                    if debug_mode:
                        text_output += f"\n**String Event**: {event}\n" # Log string event.
            elif debug_mode:
                text_output += f"\n**Event**: {event}\n" # Log other events.
            yield full_response # Yield partial response.

        # --- GitHub Issue Creation ---
        tool_calls_comment_text = f"**Tool Calls:**\n```\n{tool_calls_output}\n```" if show_tool_calls else None # Format tool calls for comment.
        debug_log_comment_text = f"**Debug Log:**\n```\n{text_output}\n```" if debug_mode else None # Format debug log for comment.
        event_data_comment_text = f"**Event Data:**\n```\n{st.session_state.event_data}\n```" if show_event_data and st.session_state.event_data else None # Format event data for comment.

        issue_created = create_github_issue(
            title=query,  # Issue title: user question.
            body=full_response, # Issue body: bot answer.
            tool_calls_comment=tool_calls_comment_text, # Tool calls comment.
            debug_log_comment=debug_log_comment_text, # Debug log comment.
            event_data_comment=event_data_comment_text # Event data comment.
        )

        if issue_created:
            logging.info("GitHub issue creation completed.") # Log issue creation success.
        else:
            logging.error("GitHub issue creation failed.") # Log issue creation failure.


        # --- Post-Stream Processing ---
        st.session_state.chat_history[latest_index]['bot'] = full_response # Update chat history with full response.
        if debug_mode:
            st.session_state.debug_output = text_output # Store debug output.
        if show_tool_calls:
            st.session_state.tool_calls_output = tool_calls_output # Store tool calls output.
        if show_event_data:
            st.session_state.event_data = event # Store last event data.

    except Exception as e: # --- Error Handling ---
        logging.error(f"Error processing response: {e}", exc_info=True) # Log response processing error.
        yield "I encountered an error processing your request. Please try again later." # User error message.

# --- Initialize Session State Variables ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # Initialize chat history (list of user/bot message dictionaries).

if 'debug_output' not in st.session_state:
    st.session_state.debug_output = "" # Initialize debug output string.

if 'tool_calls_output' not in st.session_state:
    st.session_state.tool_calls_output = "" # Initialize tool calls output string.

if 'event_data' not in st.session_state:
    st.session_state.event_data = None # Initialize event_data.

# --- Streamlit App Description ---
st.title("Agentic AI Chatbot: autonomous agent that takes actions")
#st.write("This page demonstrates an AI chatbot using different language models and tools. Enable 'Show Tool Calls' and 'Show Debug Log' in the sidebar to see the agent's workings.")

############§§§§§§§§§§§§§§§§§§§§§############

# --- Sidebar Checkboxes and Help Section ---
with st.sidebar.expander("Help & Display Options",  expanded=True):
    show_tool_calls = st.checkbox("Show Tool Calls", value=True) # Show tool call details.
    st.caption("Display tools used by chatbot.") # Description for "Show Tool Calls"

    debug_mode = st.checkbox("Show Debug Log", value=True) # Enable debug log.
    st.caption("Detailed technical logs for debugging.") # Description for "Show Debug Log"

    show_event_data = st.checkbox("Show Event Data", value=True) # Show raw event data.
    st.caption("Raw agent communication data (technical).") # Description for "Show Event Data"

# --- Display Chat History from Session State ---
for chat in st.session_state.chat_history:
    with st.chat_message("user"): # User message in chat format.
        st.write(chat['user'])
    if chat['bot']:
        with st.chat_message("assistant"): # Bot message in chat format.
            st.write(chat['bot'])

# --- User Input Handling ---
if user_input := st.chat_input("A:"):
    # Append user message to chat history
    st.session_state.chat_history.append({"user": user_input, "bot": ""})
    latest_index = len(st.session_state.chat_history) - 1

    # Display User Message in Chat
    with st.chat_message("user"):
        st.write(user_input)

    # Placeholder for Bot Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
          for response in stream_query_response(user_input, debug_mode=debug_mode, show_event_data=show_event_data, show_tool_calls=show_tool_calls):
            full_response = response
            response_placeholder.markdown(full_response)
else:
    #st.write("Start with sample questions or ask your own:") # Guide users.
    pass

    col1, col2, col3 = st.columns(3) # Columns for buttons.

    with col1:
        if st.button("Aeroplanes in Berlin sky?", key="sample_1"): # Sample question 1.
            user_input = "Can I see aeroplanes flying in Berlin sky right now taking into consideration the current weather in Berlin?"

    with col2:
        if st.button("What is Model collapse?", key="sample_2"): # Sample question 2.
            user_input = "Breifly explain concept of model collapse."

    with col3:
        if st.button("Who wins - Laptop Man vs Superman?", key="sample_3"): # Sample question 3.
            user_input = "Who is Laptop Man? Where did you find information about him? Who is Superman? Where did you find information about him? Is Laptop Man stronger than Superman?Avoid long answer."

    if 'user_input' not in locals():
        user_input = None

    if user_input:
        # Append user message to chat history
        st.session_state.chat_history.append({"user": user_input, "bot": ""})
        latest_index = len(st.session_state.chat_history) - 1

        # Display User Message in Chat
        with st.chat_message("user"):
            st.write(user_input)

        # Placeholder for Bot Response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
              for response in stream_query_response(user_input, debug_mode=debug_mode, show_event_data=show_event_data, show_tool_calls=show_tool_calls):
                full_response = response
                response_placeholder.markdown(full_response)

# --- Conditional Display of Debug and Tool Call Output Expanders ---
if show_tool_calls and st.session_state.tool_calls_output: # Show tool calls if enabled and output exists.
    with st.expander("Tool Interaction Details"):
        st.write("Tools chatbot used to respond. Shows activated tools and instructions. Helps understand chatbot workings.")
        st.code(st.session_state.tool_calls_output)

if debug_mode and st.session_state.debug_output: # Show debug log if enabled and output exists.
    with st.expander("Detailed Debugging Information"):
        st.write("Technical log of chatbot's thought process. Useful for understanding chatbot steps, internal messages. For debugging and advanced understanding.")
        st.code(st.session_state.debug_output)

if show_event_data and st.session_state.event_data: # Show event data if enabled and data exists.
    with st.expander("Raw Agent Communication Data (Technical)"):
        st.write("Raw, technical data stream from chatbot agent. Advanced debugging showing step-by-step agent communication. For developers/technical users.")
        st.write("Event Details:", st.session_state.event_data)
