import streamlit as st
from dotenv import load_dotenv
import os
import uuid
import bs4
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
#https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFDirectoryLoader.html
from langchain_community.document_loaders import PyPDFDirectoryLoader
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
from langchain.schema import AIMessage, HumanMessage
from langchain_core.messages import trim_messages

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
      #loader = PyPDFLoader(file_path="./input_files/Laptop-Man.pdf")
      loader = PyPDFDirectoryLoader(path="./input_files/")
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
    "pdf_document_retriever",
    "Retrieves and provides information from the available PDF documents.",
)

internet_search = TavilySearchResults(max_results=2)

wikipedia_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

#tools = [retriever_tool, wikipedia_search, internet_search]
tools = [retriever_tool]

# Setup memory
# Remember to add memory filtering later - https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/
memory = MemorySaver()

# Use threads
unique_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": unique_id}}

# Create agents
agent_executor_with_memory = create_react_agent(model, tools, checkpointer=memory)

# Custom prompts
custom_prompt_template = """
You are a Retrieval-Augmented Generation (RAG) chatbot. When responding to user inquiries, adhere to these guidelines:

1. **Source-Based Responses**: Respond exclusively using the information provided in the retrieved documents by the retriever_tool.

2. **Transparency in Uncertainty**: If the retrieved information does not address the user's question, respond with: "I'm sorry, but I don't have the information to answer that question." Avoid fabricating or speculating on information.  Do not generate content beyond the retrieved data.

3. **Citation of Sources**: Always cite your source for information e.g. the name of the input document that was used to retreive the information.

4. **Clarity and Conciseness**: Communicate in a clear and concise manner, ensuring that responses are easily understood by users.

5. **Neutral Tone**: Maintain a neutral and informative tone, focusing on delivering factual information without personal opinions or biases.

Use guidelines above to respond to this input from the user : {query}
"""

# Function to stream responses
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI

# Define the model
model = ChatOpenAI(model="gpt-4o-mini")

# Function to stream responses
def stream_query_response(query, debug_mode=False):
    # Get all previous messages from chat history
    previous_messages = []
    for chat in st.session_state.chat_history:
        previous_messages.append(HumanMessage(content=chat['user']))
        if chat['bot'] and isinstance(chat['bot'], (str, dict)):
            # Extract string content from bot response if it's a dict
            bot_content = chat['bot']['messages'][-1].content if isinstance(chat['bot'], dict) else chat['bot']
            previous_messages.append(AIMessage(content=str(bot_content)))

    # Add current query
    previous_messages.append(HumanMessage(content=query))

    # Trim the messages to fit within the token limit
    # https://python.langchain.com/docs/how_to/trim_messages
		# input_tokens: Number of tokens in the input messages sent to the model. This should align with your `max_tokens` setting if trimming is applied correctly.
    # output_tokens: Number of tokens in the model's response. Check out the trace in debug mode.
		# total_tokens: Sum of `input_tokens` and `output_tokens`, representing the entire interaction's token count.
		# completion_tokens: Similar to `output_tokens`, indicating tokens used for the model's generated response.

    trimmed_messages = trim_messages(
        previous_messages,
        strategy="last",
        token_counter=model,  # Use the model for token counting
        max_tokens=100,  # Adjust this to your context window size
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
        allow_partial=False,
    )

    custom_prompt = custom_prompt_template.format(query=query)
    final_response = ""

    try:
        # Stream the response from the agent with trimmed message history
        for event in agent_executor_with_memory.stream(
            {"messages": trimmed_messages},  # Use trimmed messages
            config=config,
            stream_mode="values",
        ):
            if isinstance(event, (str, dict)):
                # Extract string content if event is a dict
                final_response = event['messages'][-1].content if isinstance(event, dict) else event

                yield str(final_response)

                # Conditionally show event data
                if debug_mode:
                    with st.expander("Show Event Data"):
                        st.write("Event Details:", event)

    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        yield "I encountered an error processing your request."

# Initialize session state for chat history
# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
st.title("LangChain Chatbot with Streamlit Frontend")

# Debug mode toggle in the sidebar
st.sidebar.title("Settings")
debug_mode = st.sidebar.checkbox("Show Debug Details", value=False)

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat['user'])
    if chat['bot']:
        with st.chat_message("assistant"):
            st.write(chat['bot'])

# User input
if user_input := st.chat_input("You:"):  # Chat input replaces text_input + button
    # Add user message to chat history
    st.session_state.chat_history.append({"user": user_input, "bot": ""})
    latest_index = len(st.session_state.chat_history) - 1

    # Display the user's input
    with st.chat_message("user"):
        st.write(user_input)

    # Placeholder for bot response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        # Stream the response
        for response in stream_query_response(user_input, debug_mode=debug_mode):
            st.session_state.chat_history[latest_index]['bot'] = response
            # Update the placeholder with the latest response
            response_placeholder.markdown(response)
