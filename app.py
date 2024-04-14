import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from style_templates import css
import os

load_dotenv()
GOOGLE_API_KEY = "AIzaSyDb3vYFYQIz0CSJug8VFPdWRJ5X4GS0BfY"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    # Load Faiss index with allow_dangerous_deserialization=True
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write(f"PDF🤖BOT: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.title("Chat with PDF using Gemini-Pro🚀")

    # Toggle button for dark and light mode
    dark_mode = st.sidebar.checkbox("Dark Mode", value=True)

    # Apply custom CSS based on the toggle button
    if dark_mode:
        set_dark_mode()
    else:
        set_light_mode()

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu📃:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if pdf_docs:
            # Check if any non-PDF files are uploaded
            non_pdf_files = [file for file in pdf_docs if not file.name.endswith('.pdf')]
            if non_pdf_files:
                st.error("Error: Please upload only PDF files.")
            else:
                if st.button("Process"):
                    progress_bar = st.progress(0)
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        for percent_complete in range(100):
                            progress_bar.progress(percent_complete + 1)
                    st.success("Successfully Processed")

def set_light_mode():
    # Custom CSS for light mode
    st.markdown(css, unsafe_allow_html=True)

def set_dark_mode():
    # Custom CSS for dark mode
    st.markdown("""
        <style>
        .main {
            color: white;
            background-color: #1E1E1E;
        }
        </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
