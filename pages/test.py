import streamlit as st
from dotenv import load_dotenv
import os
import uuid
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

# Load environment variables
load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(
    page_title="ChatBot",
    page_icon=":chat-plus-outline:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API keys and other configurations
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

# Define RAG tool and load documents
@st.cache_resource
def load_documents():
      #loader = PyPDFLoader(file_path="./input_files/Laptop-Man.pdf")
    loader = PyPDFDirectoryLoader(path="./input_files/")
    return loader.load()

docs = load_documents()

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# Create vector store for document retrieval
@st.cache_resource
def create_vectorstore():
    embedding_wrapper = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={'batch_size': embedding_batch_size}
    )
    return FAISS.from_documents(documents=all_splits, embedding=embedding_wrapper)

vectorstore = create_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Define tools for retrieval and search
retriever_tool = create_retriever_tool(
    retriever,
    "pdf_document_retriever",
    "Retrieves and provides information from the available PDF documents."
)

internet_search = TavilySearchResults(max_results=2)

wikipedia_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

#tools = [retriever_tool, wikipedia_search, internet_search]
tools = [retriever_tool]

# Setup memory management for conversation history
# Remember to add memory filtering later - https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/
memory = MemorySaver()

# Create agent with memory support using LangChain's REACT framework
agent_executor_with_memory = create_react_agent(model, tools, checkpointer=memory)

# Custom prompt template for RAG chatbot responses
custom_prompt_template = """
You are a Retrieval-Augmented Generation (RAG) chatbot. When responding to user inquiries, adhere to these guidelines:
1. **Source-Based Responses**: Respond using information from retrieved documents when relevance is high.
2. **Fallback Explanation**: If relevance is low, explain that you're using alternative methods to gather information (e.g., Wikipedia or internet search).
3. **Transparency in Uncertainty**: If you cannot answer, respond with: "I'm sorry, but I don't have the information to answer that question."
4. **Citation of Sources**: Always cite your source (e.g., PDF document name, Wikipedia link).
5. **Clarity and Conciseness**: Communicate clearly and concisely.
6. **Neutral Tone**: Maintain a neutral and informative tone.

Respond to this input from the user: {query}
"""

# Dynamic tool selection based on relevance
def select_tool(query, retrieved_docs):
    if not retrieved_docs or all(doc["relevance_score"] < 0.7 for doc in retrieved_docs):
        st.info("Documents have low relevance. Switching to internet search...")
        return internet_search.run(query)
    return retriever_tool.run(query)

# Function to stream responses from LangChain agent with trimmed message history
def stream_query_response(query, debug_mode=False):
    previous_messages = []
    for chat in st.session_state.chat_history:
        previous_messages.append(HumanMessage(content=chat['user']))
        if chat['bot']:
            bot_content = chat['bot']['messages'][-1].content if isinstance(chat['bot'], dict) else chat['bot']
            previous_messages.append(AIMessage(content=str(bot_content)))

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
        token_counter=model,
        max_tokens=100,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
        allow_partial=False,
    )

    custom_prompt = custom_prompt_template.format(query=query)

    try:
        for event in agent_executor_with_memory.stream({"messages": trimmed_messages}, config={"configurable": {"thread_id": str(uuid.uuid4())}}, stream_mode="values"):
            final_response = event['messages'][-1].content if isinstance(event, dict) else event

            # Conditional edge for low relevance
            if "relevance_score" in event and event["relevance_score"] < 0.7:
                yield "The retrieved documents may not be highly relevant. Attempting additional resources..."
                secondary_tool_response = wikipedia_search.run(query)
                yield str(secondary_tool_response)
                break

            yield str(final_response)

            if debug_mode:
                with st.expander("Show Event Data"):
                    st.write("Event Details:", event)
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        yield "I encountered an error processing your request."

# Initialize session state for chat history if not already done
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat interface with Streamlit components
st.title("LangChain Chatbot with Streamlit Frontend")

debug_mode = st.sidebar.checkbox("Show Debug Details", value=False)

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat['user'])
    if chat['bot']:
        with st.chat_message("assistant"):
            st.write(chat['bot'])

if user_input := st.chat_input("You:"):
    st.session_state.chat_history.append({"user": user_input, "bot": ""})
    latest_index = len(st.session_state.chat_history) - 1

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        for response in stream_query_response(user_input, debug_mode=debug_mode):
            st.session_state.chat_history[latest_index]['bot'] = response
            response_placeholder.markdown(response)
