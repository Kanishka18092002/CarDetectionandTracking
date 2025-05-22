import json
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings # HuggingFace used here
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

conversation_chain = None


def create_vectorstore_from_analytics(analytics: dict):
    """
        Converts analytics dict to a FAISS vector store using HuggingFaceEmbeddings.
        - FAISS for fast, in-memory similarity search.
        - HuggingFaceEmbeddings for flexible,cost free embeddings.
    """
    #  Convert analytics JSON data into a formatted text string.
    analytics_text = json.dumps(analytics, indent=2)
    # Wrap the text into a Document for Langchain processing.
    docs = [Document(page_content=analytics_text, metadata={"source": "analytics"})]
    # You can customize the model inside by default it is sentence-transformers/all-MiniLM-L6-v2
    embeddings = HuggingFaceEmbeddings()# Generate vector embeddings (numbers) from the text using HuggingFace.
    #Store these vectors in a FAISS database for fast similarity search.
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

def init_conversation_chain(analytics: dict):
    """
    Initializes the Conversational Retrieval Chain:
    - Loads embeddings into FAISS
    - Uses Ollama's local LLM (e.g., mistral)
    - Stores conversation history to allow follow-up questions
    """
    global conversation_chain
    vectorstore = create_vectorstore_from_analytics(analytics)
     # Locally hosted LLM and when temp is  higher means more creative answers, lower means more deterministic
    llm = OllamaLLM(model="mistral", temperature=0.7)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # Stores context
    retriever = vectorstore.as_retriever()  # The retriever is responsible for searching the vector store
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

def query_ollama(analytics: dict, user_question: str) -> str:
    """
       Main function to handle user queries:
        - Initializes the chain if not done yet
        - Feeds the user's question into the chain
        - Returns the model's answer based on context + analytics
    """
    global conversation_chain
    if conversation_chain is None:
        init_conversation_chain(analytics)
    result = conversation_chain.invoke({"question": user_question})
    return result["answer"]








