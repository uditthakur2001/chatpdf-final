import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import pandas as pd
import docx
import openpyxl
import os

# Set the Google API Key in the environment
st.set_page_config(page_title="ChatPDF", page_icon="📝", layout="wide")

# st.title("📝 Document QA App with Generative AI")
st.title("📝 ChatPDF")

st.write("Upload a PDF, Excel, Word, CSV, or Text document, ask a question and get accurate & detailed responses.")

# Input: Google API Key
api_key = "AIzaSyCN6sMI7nRyuwgTovMrNbs2OdoOIEFAv60"

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# File Upload Section
st.sidebar.header("📂 Upload a Document")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF, Excel, Word, CSV, or Text file:",
    type=["pdf", "xlsx", "xls", "docx", "csv", "txt"],
)

# User Question Section
st.sidebar.header("📝 Ask a Question")
user_question = st.sidebar.text_area(
    "Enter your question:",
    placeholder="E.g., What is the main topic of the document?",
)

# Add a Search Button
st.sidebar.button("🔍 Search")

# # File to store question-answer history
# qa_history_file = "qa_history.txt"

# def save_to_txt(question, answer):
#     """Save the question and answer to a text file."""
#     with open(qa_history_file, "a", encoding="utf-8") as f:
#         f.write(f"Question: {question}\n")
#         f.write(f"Answer: {answer}\n")
#         f.write("=" * 50 + "\n")  # Separator for readability

@st.cache_data
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
    return text

@st.cache_data
def extract_text_from_excel(excel_file):
    # df = pd.read_excel(excel_file)
    df = pd.read_excel(excel_file, engine="openpyxl")
    return "\n".join(df.astype(str).values.flatten())

@st.cache_data
def extract_text_from_word(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

@st.cache_data
def extract_text_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return "\n".join(df.astype(str).values.flatten())

@st.cache_data
def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say, "Answer is not available in the context." Don't provide a wrong answer.
    
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Process Uploaded File
if uploaded_file and api_key and user_question:
    with st.spinner("Processing your document..."):
        try:
            file_type = uploaded_file.name.split(".")[-1]
            if file_type == "pdf":
                raw_text = extract_text_from_pdf(uploaded_file)
            elif file_type in ["xlsx", "xls"]:
                raw_text = extract_text_from_excel(uploaded_file)
            elif file_type == "docx":
                raw_text = extract_text_from_word(uploaded_file)
            elif file_type == "csv":
                raw_text = extract_text_from_csv(uploaded_file)
            elif file_type == "txt":
                raw_text = extract_text_from_txt(uploaded_file)
            else:
                st.error("Unsupported file format.")
                st.stop()

            # Split text into chunks
            text_chunks = get_text_chunks(raw_text)

            # Embed text into a vector store
            vector_store = get_vector_store(text_chunks)

            # Perform similarity search and generate the answer
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

            # Extract the generated answer
            generated_answer = response["output_text"]

            # Display the results
            st.success("Answer Generated!")
            st.subheader("Your Question:")
            st.write(user_question)

            st.subheader("Generated Answer:")
            for line in generated_answer.split("\n"):
                if line.strip():
                    st.write(line)

            # # Save the question and answer to the text file
            # save_to_txt(user_question, generated_answer)

            # st.sidebar.success("Question and Answer saved to 'qa_history.txt'!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    if not api_key:
        st.warning("Please provide your Google API key.")
    if not uploaded_file:
        st.warning("Please upload a document.")
    if not user_question:
        st.warning("Please enter a question.")

# Footer
st.markdown("""
---
🌟 Powered by LangChain and Google Generative AI

Made by Udit
""")
