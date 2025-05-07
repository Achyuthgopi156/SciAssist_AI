import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import SentenceTransformer, util
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from typing import List, Dict, Any, Optional
from langchain_core.outputs import ChatResult, ChatGeneration

# HTML templates
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

class GroqLLM(BaseChatModel):
    client: Any = None
    model_name: str = "llama3-70b-8192"
    
    def __init__(self, api_key: str, model_name: str = "llama3-70b-8192"):
        super().__init__()
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        
    @property
    def _llm_type(self) -> str:
        return "groq"
        
    def _generate(
        self,
        messages: List[Dict[str, Any]],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            groq_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    groq_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, HumanMessage):
                    groq_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    groq_messages.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, dict):
                    groq_messages.append(msg)
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=groq_messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1024),
                top_p=kwargs.get("top_p", 1),
                stream=False,
                stop=stop,
            )
            
            message = completion.choices[0].message
            generation = ChatGeneration(
                message=AIMessage(content=message.content)
            )
            return ChatResult(generations=[generation])
        except Exception as e:
            st.error(f"Error calling Groq API: {str(e)}")
            raise
            
    async def _agenerate(
        self,
        messages: List[Dict[str, Any]],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, **kwargs)

def prepare_and_split_docs(pdf_directory):
    split_docs = []
    for pdf in pdf_directory:
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())
        
        loader = PyPDFLoader(pdf.name)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512,
            chunk_overlap=256,
            disallowed_special=(),
            separators=["\n\n", "\n", " "]
        )
        split_docs.extend(splitter.split_documents(documents))
    return split_docs

def ingest_into_vectordb(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    db = FAISS.from_documents(split_docs, embeddings)
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    return db

def get_conversation_chain(retriever):
    llm = GroqLLM(api_key="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")
    
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\
    
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

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

def calculate_similarity_score(answer: str, context_docs: list) -> float:
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    context_texts = [doc.page_content for doc in context_docs]
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    context_embeddings = model.encode(context_texts, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(answer_embedding, context_embeddings)
    return similarities.max().item()

# Initialize Streamlit app
st.title("Sciassist :books:")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversational_chain' not in st.session_state:
    st.session_state.conversational_chain = None
if 'show_docs' not in st.session_state:
    st.session_state.show_docs = {}
if 'similarity_scores' not in st.session_state:
    st.session_state.similarity_scores = {}

# Sidebar for file upload
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Documents", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("Process PDFs"):
        with st.spinner("Processing documents..."):
            try:
                split_docs = prepare_and_split_docs(uploaded_files)
                vector_db = ingest_into_vectordb(split_docs)
                retriever = vector_db.as_retriever()
                st.session_state.conversational_chain = get_conversation_chain(retriever)
                st.sidebar.success("Documents processed successfully!")
            except Exception as e:
                st.sidebar.error(f"Error processing documents: {str(e)}")

# Chat interface
user_input = st.chat_input("Ask a question about the documents:")

if user_input and st.session_state.conversational_chain:
    with st.spinner("Thinking..."):
        try:
            session_id = "default_session"
            response = st.session_state.conversational_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            st.session_state.chat_history.append({
                "user": user_input,
                "bot": response['answer'],
                "context_docs": response.get('context', [])
            })
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Display chat history
for index, message in enumerate(st.session_state.chat_history):
    st.markdown(user_template.format(msg=message["user"]), unsafe_allow_html=True)
    st.markdown(bot_template.format(msg=message["bot"]), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"Show/Hide Source Docs {index+1}", key=f"toggle_{index}"):
            st.session_state.show_docs[index] = not st.session_state.show_docs.get(index, False)
    
    with col2:
        if st.button(f"Calculate Relevancy {index+1}", key=f"relevancy_{index}"):
            if index not in st.session_state.similarity_scores:
                try:
                    score = calculate_similarity_score(
                        message['bot'], 
                        message['context_docs']
                    )
                    st.session_state.similarity_scores[index] = score
                except Exception as e:
                    st.error(f"Error calculating similarity: {str(e)}")
    
    if st.session_state.show_docs.get(index, False):
        with st.expander(f"Source Documents for Q&A {index+1}"):
            for doc in message.get('context_docs', []):
                st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.write(doc.page_content)
                st.divider()
    
    if index in st.session_state.similarity_scores:
        st.write(f"**Answer Relevancy Score:** {st.session_state.similarity_scores[index]:.2f}")
    
    st.divider()
