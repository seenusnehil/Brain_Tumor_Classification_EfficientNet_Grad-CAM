import streamlit as st
import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceTextGenInference
import time
import base64

# Set page configuration
st.set_page_config(
    page_title="Medical Report Assistant",
    page_icon="🩺",
    layout="wide"
)

# Apply blue-green theme with CSS
st.markdown("""
<style>
    /* Main theme colors */
    :root {{
        --primary-color: #2E86AB;
        --secondary-color: #5FB0B7;
        --background-color: #f0f5f5;
        --light-color: #e6f3f5;
        --dark-color: #184759;
        --user-msg-bg: #2E86AB;
        --bot-msg-bg: #5FB0B7;
    }}
    
    /* Set background color */
    .stApp {{
        background-color: var(--background-color);
    }}
    
    /* Header styling */
    .main-header {{
        background-color: var(--primary-color);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }}
    
    /* Chat container */
    .chat-container {{
        background-color: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        height: 400px;
        overflow-y: auto;
    }}
    
    /* Message bubbles */
    .user-message {{
        background-color: var(--user-msg-bg);
        color: white;
        border-radius: 18px 18px 0 18px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
        clear: both;
    }}
    
    .bot-message {{
        background-color: var(--bot-msg-bg);
        color: white;
        border-radius: 18px 18px 18px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        float: left;
        clear: both;
    }}
    
    /* Text input customization */
    .stTextInput div[data-baseweb="input"] {{
        border-radius: 20px;
        border: 2px solid var(--secondary-color);
    }}
    
    /* Button customization */
    .stButton button {{
        background-color: var(--primary-color);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.3rem 1rem;
        font-weight: bold;
    }}
    
    .stButton button:hover {{
        background-color: var(--dark-color);
    }}

    /* Upload button customization */
    .stUploadButton > button {{
        background-color: var(--secondary-color);
        color: white;
    }}
    
    /* Spacer for messages layout */
    .message-spacer {{
        height: 5px;
        clear: both;
    }}
    
    /* File upload and info box */
    .file-info-box {{
        background-color: var(--light-color);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid var(--primary-color);
    }}
    
    /* Fix for expander and selectbox */
    .streamlit-expanderHeader, .stSelectbox label {{
        color: var(--dark-color);
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None

# Function to create conversation chain based on document
def create_conversation_chain(pdf_docs):
    # Create temporary directory for PDFs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save PDFs to temp directory
        pdf_paths = []
        for pdf_doc in pdf_docs:
            pdf_path = os.path.join(temp_dir, pdf_doc.name)
            with open(pdf_path, 'wb') as f:
                f.write(pdf_doc.getvalue())
            pdf_paths.append(pdf_path)
        
        # Load PDF documents
        documents = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        doc_chunks = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector store
        vector_store = FAISS.from_documents(doc_chunks, embeddings)
        
        # Setup memory
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        
        # Initialize LLM
        llm = HuggingFaceTextGenInference(
            inference_server_url="https://api-inference.huggingface.co/models/BioMistral/BioMedical-MultiModal-Llama-3-8B-V1",
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.1,
            repetition_penalty=1.03
        )
        
        # Create conversation chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            chain_type="stuff",
            verbose=True
        )
        
        return conversation_chain

# Function to display chat messages
def display_chat_messages():
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">{msg["content"]}</div><div class="message-spacer"></div>', 
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{msg["content"]}</div><div class="message-spacer"></div>', 
                        unsafe_allow_html=True)

# Function to handle user input
def handle_user_input():
    user_input = st.session_state.user_input
    if user_input and st.session_state.conversation_chain:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get response from conversation chain
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversation_chain({"question": user_input})
                bot_response = response["answer"]
            except Exception as e:
                bot_response = f"I'm having trouble connecting to the BioMedical model. Please try again in a moment. Error: {str(e)}"
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        
        # Clear input box
        st.session_state.user_input = ""

# Main app header
st.markdown('<div class="main-header"><h1>🩺 Medical Report Assistant</h1></div>', unsafe_allow_html=True)

# App layout with two columns
col1, col2 = st.columns([1, 3])

with col1:
    # File upload section
    st.markdown('<h3>Upload Medical Reports</h3>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF medical reports",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    # Process button
    if st.button("Process Reports"):
        if uploaded_files:
            with st.spinner("Processing medical reports..."):
                try:
                    # Create conversation chain
                    st.session_state.conversation_chain = create_conversation_chain(uploaded_files)
                    
                    # Store processed file names
                    st.session_state.processed_files = [file.name for file in uploaded_files]
                    
                    # Add welcome message to chat history
                    if not st.session_state.chat_history:
                        welcome_msg = "Hello! I'm your Medical Report Assistant. I can answer questions about the medical reports you've uploaded. How can I help you today?"
                        st.session_state.chat_history.append({"role": "bot", "content": welcome_msg})
                    
                    st.success(f"Successfully processed {len(uploaded_files)} medical reports!")
                except Exception as e:
                    st.error(f"Error processing reports: {str(e)}")
        else:
            st.warning("Please upload at least one PDF medical report.")
    
    # Display processed files
    if st.session_state.processed_files:
        st.markdown('<div class="file-info-box">', unsafe_allow_html=True)
        st.markdown("### Active Reports")
        for file_name in st.session_state.processed_files:
            st.markdown(f"✓ {file_name}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        if st.session_state.conversation_chain:
            welcome_msg = "Chat history cleared. How else can I help you with the medical reports?"
            st.session_state.chat_history.append({"role": "bot", "content": welcome_msg})

with col2:
    # Chat interface
    st.markdown('<h3>Chat with the Medical Assistant</h3>', unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
        # Display messages
        display_chat_messages()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    st.text_input(
        "Ask a question about your medical report:",
        key="user_input",
        on_change=handle_user_input,
        disabled=not st.session_state.conversation_chain
    )
    
    # Instructions/guidance
    with st.expander("📝 Suggested Questions"):
        st.markdown("""
        Try asking questions like:
        - What is the diagnosis in the report?
        - What is the confidence level of the detection?
        - Are there any areas of concern highlighted in the report?
        - What recommendations are provided in the report?
        - Can you explain the GradCAM visualization?
        - What is the patient's information?
        - When was this report generated?
        """)

# Add autoscroll JavaScript
st.markdown("""
<script>
    // Function to scroll chat to bottom
    function scrollChatToBottom() {
        var chatContainer = document.querySelector('#chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    
    // Run on page load and when content changes
    scrollChatToBottom();
    const observer = new MutationObserver(scrollChatToBottom);
    const chatContainer = document.querySelector('#chat-container');
    if (chatContainer) {
        observer.observe(chatContainer, { childList: true, subtree: true });
    }
</script>
""", unsafe_allow_html=True)

# Footer
st.markdown('---')
st.caption('Medical Report Assistant is powered by BioMedical-MultiModal-Llama-3-8B-V1')