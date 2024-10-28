import os
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

import chromadb
chroma_client = chromadb.Client()


# Access the API key from environment variables
load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("API key for Groq is not set. Please check your .env file or environment variables.")

# Initialize the Groq client with the API key
client = Groq(api_key=api_key)

# Streamlit UI
st.title("Iskobot")

st.write("---")

# Load PDF and prepare documents for retrieval
pdf_loader = PyPDFLoader("./pdfs/1706.03762v7.pdf")
pages = pdf_loader.load_and_split()

# Split the PDF content into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)

# Generate embeddings and create a vector store for retrieval
embeddings = OllamaEmbeddings(model='jina/jina-embeddings-v2-base-en')  # Specify embedding model here

# Chroma in In-Memory Mode via persist_directory=None
vector_index = Chroma.from_texts(texts, embeddings, persist_directory=None).as_retriever(search_kwargs={"k": 2})

# Initialize session state for chat history if not already done
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display the entire chat history (excluding the latest, already shown)
for i, chat in enumerate(st.session_state.chat_history):
    st.write(f"**You:** {chat['question']}")
    st.write(f"**Iskobot:** {chat['answer']}")
    st.write("---")

# Streamlit input and button
prompt = st.text_area(label="Message Iskobot", label_visibility="hidden", placeholder="Message Iskobot")

# Button to trigger the QA process
if st.button("Ask"):
    if prompt:
        with st.spinner("Answering your question..."):
            # Use the vector store to retrieve relevant chunks
            retrieved_docs = vector_index.get_relevant_documents(prompt)
            context_from_retrieved_docs = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Define the prompt to include the retrieved context and user's question
            groq_prompt = f"""You are Iskobot, a chatbot designed to assist students with their general and academic questions.
            Use the context provided to answer the question below. If you don't know the answer, clearly state that you don't know, and avoid guessing. Provide a concise response and include 'Thanks for asking!' at the end of your answer.
            
            Context: {context_from_retrieved_docs}

            Question: {prompt}
            Answer:"""

            # Send the prompt to Groq's API
            completion = client.chat.completions.create(
                model="llama-3.2-90b-text-preview",
                messages=[{
                    "role": "user",
                    "content": groq_prompt
                }],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=True,
                stop=None,
            )

            # Collect the response from Groq and display it
            answer = ""
            for chunk in completion:
                answer += chunk.choices[0].delta.content or ""

            # Save the question and answer to session state
            st.session_state.chat_history.append({"question": prompt, "answer": answer})
            
            # Display the answer
            st.write(f"**You:** {prompt}")
            st.write(f"**Iskobot:** {answer}")