import streamlit as st
from dotenv import load_dotenv
import os
import uuid
import bs4
from langchain_openai import ChatOpenAI
#https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFDirectoryLoader.html
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
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
from langchain.schema import SystemMessage
from langchain_community.retrievers import WikipediaRetriever

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
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
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
retriever_object = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Define tools
retriever_tool = create_retriever_tool(
    retriever_object,
    "pdf_document_retriever",
    "Retrieves and provides information from the input documents.",
)

internet_search_tool = TavilySearchResults(
    max_results=2,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

#wikipedia_search_tool = WikipediaQueryRun(
#    api_wrapper=WikipediaAPIWrapper()
#)

# WikipediaRetriever is designed for retrieving documents to be used in downstream tasks within a pipeline, while WikipediaQueryRun is intended for direct querying of Wikipedia, often within agent-based frameworks.

wikipedia_retriever = WikipediaRetriever()

wikipedia_retriever_tool = create_retriever_tool(
    wikipedia_retriever,
    "wikipedia_retriever",
    "Retrieves and provides information from Wikipedia articles.",
)

#tools = [retriever_tool, wikipedia_search_tool, internet_search_tool]
tools = [retriever_tool, wikipedia_retriever_tool, internet_search_tool]

# Setup memory
# Remember to add memory filtering later - https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/
memory = MemorySaver()

# Use threads
unique_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": unique_id}}

# Create agents
agent_executor_with_memory = create_react_agent(model, tools, checkpointer=memory)

# Custom prompts using LangChain's PromptTemplate
custom_prompt_template = PromptTemplate(
    template="""You are an AI assistant equipped with the following tools:
- **retriever_tool**: This is a Retrieval Augmented Generation Tool that retrieves information from input documents.
- **wikipedia_search_tool**: Fetches information from Wikipedia articles based.
- **internet_search_tool**: Conducts real-time internet searches for current information using Tavily Search.

When answering queries posed by the user, follow this sequence:
1. First check for answer to user's query by invoking the retriever_tool
2. If no answer is found within the information retrieved by the retriever_tool, then invoke the wikipedia_search_tool and seek answer to the user's query.
3. If no answer is found within the information retrieved by the retriever_tool and then the wikipedia_search_tool, then invoke the internet_search_tool and seek answer to the user's query.

If the retrieved information from any of these tools does not address the user's question, respond with: "I'm sorry, but I don't have the information to answer that question."
Avoid fabricating or speculating on information. Do not generate content beyond the retrieved data. Always cite your source for information e.g. the name of the input document that was used to retrieve the information.

Here is the input from the user: {query}""",
    input_variables=["query"]
)

# Function to stream responses
def stream_query_response(query, debug_mode=False):
    # Initialize previous messages with the custom prompt as a system message
    previous_messages = [SystemMessage(content=custom_prompt_template.format(query=query))]

    # Append the chat history
    for chat in st.session_state.chat_history:
        previous_messages.append(HumanMessage(content=chat['user']))
        if chat['bot'] and isinstance(chat['bot'], (str, dict)):
            bot_content = chat['bot']['messages'][-1].content if isinstance(chat['bot'], dict) else chat['bot']
            previous_messages.append(AIMessage(content=str(bot_content)))

    # Add current query
    previous_messages.append(HumanMessage(content=query))

    # Trim messages
    trimmed_messages = trim_messages(
        previous_messages,
        strategy="last",
        token_counter=model,
        max_tokens=2000,
        start_on="system",
        end_on=("human", "tool"),
        include_system=True,
        allow_partial=False,
    )

    final_response = ""
    try:
        # Stream the response from the agent with trimmed message history
        for event in agent_executor_with_memory.stream(
            {"messages": trimmed_messages},
            config=config,
            stream_mode="values",
        ):
            if isinstance(event, (str, dict)):
                final_response = event['messages'][-1].content if isinstance(event, dict) else event
            yield str(final_response)
            if debug_mode:
                with st.expander("Show Event Data"):
                    st.write("Event Details:", event)
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        yield "I encountered an error processing your request."

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

        # Stream the response
        for response in stream_query_response(user_input, debug_mode=debug_mode):
            st.session_state.chat_history[latest_index]['bot'] = response
            # Update the placeholder with the latest response
            response_placeholder.markdown(response)
