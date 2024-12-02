# LangChain-MultiTool-Agent

**LangChain-MultiTool-Agent** is an advanced AI chatbot that uses LangChain agents and tools to provide intelligent, context-aware responses.

It dynamically decides which tool to invoke - be it a retrieval system, web search, or Wikipedia query - depending on the user's question. This ensures precise and relevant answers every time.

## Features

- **Dynamic Tool Selection**: The agent intelligently decides which tool to use based on the query.
- **Retrieval-Augmented Generation (RAG)**: Accesses a custom knowledge base with FAISS-backed retrieval for accurate responses.
- **Web Search Integration**: Searches the internet in real-time for up-to-date answers.
- **Wikipedia Integration**: Queries Wikipedia for factual and detailed information.
- **Persistent Memory**: Maintains session context to provide continuity across conversations.

## Tech Stack

- **[LangChain](https://langchain.com/)**: Framework for building language model applications.
- **FAISS**: For efficient vector search and document retrieval.
- **HuggingFace Embeddings**: Semantic similarity for knowledge retrieval.
- **Streamlit**: Interactive frontend for user-friendly interactions.

## LangChain Tools Used

1. **RAG Tool**: Retrieves information from a pre-defined knowledge base.
2. **Search Tool**: Fetches results from the web.
3. **Wikipedia Tool**: Pulls data from Wikipedia.

## Use Cases

- Quick fact-checking and research.
- Summarizing complex topics.
- Answering domain-specific questions with retrieval capabilities.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/kanad13/LangChain-MultiTool-Agent.git
   cd LangChain-MultiTool-Agent
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   Create a `.env` file and populate it with your API keys:

   ```env
   OPENAI_API_KEY=your_openai_key
   TAVILY_API_KEY=your_tavily_key
   LANGCHAIN_API_KEY=your_langchain_key
   ```

4. Run the app locally:

   ```bash
   streamlit run app.py
   ```

5. Open your browser to the URL provided by Streamlit.

## 🔧 Configuration

- **Customizing Tools**: Modify the `tools` list in `app.py` to add or remove tools.
- **Knowledge Base**: Update the FAISS index with your own documents for domain-specific applications.

## How It Works

1. **Query Input**: The user submits a query through the Streamlit frontend.
2. **Agent Decision**: The LangChain agent determines the most relevant tool to handle the query.
3. **Tool Execution**: The selected tool fetches the required data.
4. **Response Generation**: The agent combines tool output with reasoning to generate the final response.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
