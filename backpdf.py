import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        except Exception as e:
            st.error(f"Error processing PDF {pdf.name if hasattr(pdf, 'name') else 'file'}: {str(e)}")
    return text.strip()

# Function to split text into smaller, more focused chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Reduced chunk size for more precise context
        chunk_overlap=200,  # Reduced overlap to avoid redundancy
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter out very short chunks

# Function to create and save vector store
def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No valid text chunks to process")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Specify a more recent embedding model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to set up conversational chain with a refined prompt
def get_conversational_chain():
    prompt_template = """
    Provide a detailed and accurate answer to the question based solely on the provided context from the PDF documents. 
    Structure the response in a clear, engaging, and user-friendly format. 
    If the answer is not found in the context, state: "Answer is not available in the context."

    ### Context:
    {context}

    ### Question:
    {question}

    ### Answer:
    """
    
    model = ChatOpenAI(model_name="gpt-4", temperature=0.1)  # Lower temperature for more precise answers
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input and generate response
def user_input(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    try:
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        # Increase k to get more relevant documents
        docs = vector_store.similarity_search(user_question, k=5)
        
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        return response["output_text"]
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return "Unable to process the question due to an error."

# Main Streamlit app
def main():
    st.set_page_config(page_title="Yoga Bot", layout="wide")
    st.header("Yoga Chatbot using OpenAI 💁")

    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False

    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Upload Your PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(uploaded_files)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.processed = True
                    st.success(f"Processed {len(uploaded_files)} PDFs successfully!")
                else:
                    st.error("No text could be extracted from the PDFs.")

    # Main chat interface
    if st.session_state.processed:
        st.subheader("Ask Your Question")
        user_question = st.text_input("Ask a question about your PDFs:")
        
        if user_question:
            with st.spinner("Searching for answers..."):
                response = user_input(user_question)
                st.markdown("### Response")
                st.write(response)
    elif uploaded_files:
        st.info("Please click 'Process PDFs' to begin.")
    else:
        st.info("Please upload PDF files to start chatting.")

if __name__ == "__main__":
    main()