import streamlit as st
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import SentenceTransformer, util
#from htmlTemplate import css, bot_template, user_template
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import fitz  # PyMuPDF for PDF processing
import io
import torch


bot_template = '''
<div style="display: flex; align-items: center; margin-bottom: 10px;">
    <div style="flex-shrink: 0; margin-right: 10px;">
        <img src="https://uxwing.com/wp-content/themes/uxwing/download/communication-chat-call/answer-icon.png" 
             style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div style="background-color:black; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
        {msg}
    </div>
</div>
'''

user_template = '''
<div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: flex-end;">
    <div style="flex-shrink: 0; margin-left: 10px;">
        <img src="https://cdn.iconscout.com/icon/free/png-512/free-q-characters-character-alphabet-letter-36051.png?f=webp&w=512" 
             style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div style="background-color: #007bff; color: black; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
        {msg}
    </div>
</div>
'''

button_style = """
<style>
    .small-button {
        display: inline-block;
        padding: 5px 10px;
        font-size: 12px;
        color: white;
        background-color: #007bff;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin-right: 5px;
    }
    .small-button:hover {
        background-color: #0056b3;
    }
</style>
"""



# Initialize CLIP for visual embedding
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to prepare and split documents, now including images
def prepare_and_split_docs(pdf_directory):
    split_docs = []
    for pdf in pdf_directory:
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())
        
        # Use PyPDFLoader with the saved file path
        loader = PyPDFLoader(pdf.name)
        documents = loader.load()

        # Extract images from each PDF page
        images = extract_images_from_pdf(pdf.name)

        # Process and split text content
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512,
            chunk_overlap=256,
            disallowed_special=(),
            separators=["\n\n", "\n", " "]
        )
        split_docs.extend(splitter.split_documents(documents))

        # Add images to split_docs as individual "documents"
        for img in images:
            split_docs.append({"content": img, "type": "image"})

    return split_docs

# Extract images from a PDF
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    return images

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Function to ingest documents into the vector database
def ingest_into_vectordb(split_docs):
    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    
    # Ingest documents into FAISS vector store directly with embeddings
    db = FAISS.from_documents(split_docs, embeddings)
    
    # Save the FAISS vector store locally
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    return db


# Function to create a visual embedding for an image
def create_visual_embedding(image: Image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings[0].numpy()

# Function to get the conversation chain (same as before)
def get_conversation_chain(retriever):
    llm = Ollama(model="llama3.2")
    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, "
        "provide a response that directly addresses the user's query based on the provided documents. "
        "Do not rephrase the question or ask follow-up questions."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    system_prompt = (
        """As a personal chat assistant, You are programmed to deliver precise and pertinent information strictly 
        based on the provided documents. Each response will be concise, limited to 150 words and formulated in 2-3 sentences, 
        without any options for selection, standalone questions, or queries included. use this {context}"""
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    


    ### Statefully manage chat history ###
    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

from langchain_huggingface import HuggingFaceEmbeddings
import torch

def calculate_similarity_score(answer: str, context_docs: list) -> float:
    model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    context_docs = [doc.page_content for doc in context_docs]
    
    # Use `embed_query` and `embed_documents` for HuggingFaceEmbeddings
    answer_embedding = torch.tensor(model.embed_query(answer))
    context_embeddings = torch.tensor(model.embed_documents(context_docs))

    # Calculate cosine similarities
    similarities = util.pytorch_cos_sim(answer_embedding, context_embeddings)

    # Return the maximum similarity score from the context documents
    max_score = similarities.max().item()
    return max_score


st.title("Sciassist :books:")

# Sidebar for file upload
uploaded_files = st.sidebar.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.sidebar.button("Process PDFs"):
        split_docs = prepare_and_split_docs(uploaded_files)
        vector_db = ingest_into_vectordb(split_docs)
        retriever = vector_db.as_retriever()
        st.sidebar.success("Documents processed and vector database created!")

        # Initialize the conversation chain
        conversational_chain = get_conversation_chain(retriever)
        st.session_state.conversational_chain = conversational_chain

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_docs' not in st.session_state:
    st.session_state.show_docs = {}

if 'similarity_scores' not in st.session_state:
    st.session_state.similarity_scores = {}

# Function to toggle the document visibility
def toggle_docs(index):
    st.session_state.show_docs[index] = not st.session_state.show_docs.get(index, False)

# Chat input
user_input = st.text_input("Ask a question about the documents:")

if st.button("Submit"):
    st.markdown(button_style, unsafe_allow_html=True)
    if user_input and 'conversational_chain' in st.session_state:
        session_id = "abc123"  # Static session ID for this demo; you can make it dynamic if needed
        conversational_chain = st.session_state.conversational_chain
        response = conversational_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        context_docs = response.get('context', [])
        st.session_state.chat_history.append({"user": user_input, "bot": response['answer'],  "context_docs": context_docs})

# Display chat history
if st.session_state.chat_history:
    for index, message in enumerate(st.session_state.chat_history):
        # Render the user message using the template
        st.markdown(user_template.format(msg=message['user']), unsafe_allow_html=True)
        
        # Render the bot message using the bot template
        st.markdown(bot_template.format(msg=message['bot']), unsafe_allow_html=True)

        # Initialize session state for each message
        if f"show_docs_{index}" not in st.session_state:
            st.session_state[f"show_docs_{index}"] = False
        if f"similarity_score_{index}" not in st.session_state:
            st.session_state[f"similarity_score_{index}"] = None

        # Layout for the buttons in a single row (horizontal alignment)
        cols = st.columns([1, 1])  # Create two equal columns for buttons

        # Render "Show Source Docs" button
        with cols[0]:
            if st.button(f"Show/Hide Source Docs", key=f"toggle_{index}"):
                # Toggle the visibility of source documents for this message
                st.session_state[f"show_docs_{index}"] = not st.session_state[f"show_docs_{index}"]

        # Render "Answer Relevancy" button
        with cols[1]:
            if st.button(f"Calculate Answer Relevancy", key=f"relevancy_{index}"):
                if st.session_state[f"similarity_score_{index}"] is None:
                    score = calculate_similarity_score(message['bot'], message['context_docs'])
                    st.session_state[f"similarity_score_{index}"] = score

        # Check if source documents should be shown
        if st.session_state[f"show_docs_{index}"]:
            with st.expander("Source Documents"):
                for doc in message.get('context_docs', []):
                    st.write(f"Source: {doc.metadata['source']}")
                    st.write(doc.page_content)

        # Display similarity score if available
        if st.session_state[f"similarity_score_{index}"] is not None:
            st.write(f"Similarity Score: {st.session_state[f'similarity_score_{index}']:.2f}")
