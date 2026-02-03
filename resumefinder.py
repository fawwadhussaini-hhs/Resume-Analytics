import os
import io
import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader

# LangChain & Gemini Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION & AUTH ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
# Recommendation: Move this to streamlit secrets for security
os.environ["GOOGLE_API_KEY"] = "AIzaSyAaJURrelGrjwoDc87I6q3qOgo8Z8ohlT8"
FOLDER_ID = "1keH1ZBi5B9P_vRLuHQ5Ch8a8ECSzxW8EL6uYQA7t08VxTx1BfxitXDe9lfSQNbfcUGBqtVqG"


def get_drive_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)


# --- 2. DATA INGESTION ---
def process_files_to_chroma(files):
    service = get_drive_service()
    documents = []
    for file in files:
        if file['mimeType'] == 'application/pdf':
            request = service.files().get_media(fileId=file['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            fh.seek(0)
            reader = PdfReader(fh)
            text = "".join([page.extract_text() for page in reader.pages])
            documents.append(Document(page_content=text, metadata={"source": file['name']}))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_db


def get_resume_count():
    """Fetches the count of PDFs from Drive."""
    try:
        service = get_drive_service()
        query = f"'{FOLDER_ID}' in parents and mimeType = 'application/pdf' and trashed = false"
        results = service.files().list(q=query, fields="files(id)").execute()
        return len(results.get('files', []))
    except Exception as e:
        return 0


# --- 3. RAG BRAIN ---
def get_llm_analysis(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs = vector_db.similarity_search(query, k=7)

    context = "\n\n".join([f"FILENAME: {d.metadata['source']}\nCONTENT: {d.page_content}" for d in docs])

    system_prompt = """
    You are a professional talent hunter. Analyze the resumes to find the best fit.

    CONTEXT FROM RESUMES:
    {context}

    USER REQUIREMENT:
    {question}

    RESPONSE FORMAT:
    1. Identify top candidates.
    2. Provide a 'Match Score' (0-100).
    3. List key skills.
    4. Contact details.
    """

    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})
    return response.content, docs


# --- 4. STREAMLIT UI (Cleaned & Consolidated) ---
st.set_page_config(page_title="Resume AI Finder", layout="wide")
st.title("📂 Resume AI Analysis Engine")

# Fetch count for the header
resume_count = get_resume_count()
st.metric("Resumes in Drive Folder", resume_count)

# Sidebar with a single button
with st.sidebar:
    st.header("Settings")
    # Added a 'key' to ensure uniqueness just in case
    if st.button("🔄 Refresh & Index Drive Folder", key="sync_button"):
        service = get_drive_service()
        query = f"'{FOLDER_ID}' in parents and trashed = false"
        files = service.files().list(q=query, fields="files(id, name, mimeType)").execute().get('files', [])

        if files:
            with st.spinner("Processing Resumes..."):
                process_files_to_chroma(files)
            st.success("Indexing Complete!")
            st.rerun()  # Refresh to update the metric count
        else:
            st.warning("No files found in folder.")

# Main Search Interface
user_query = st.text_input("Describe your ideal candidate (e.g., 'Senior Python Developer with AWS'):")

if user_query:
    if os.path.exists("./chroma_db"):
        with st.spinner("Analyzing resumes..."):
            answer, sources = get_llm_analysis(user_query)

        tab1, tab2 = st.tabs(["📊 AI Recommendation", "🔍 Source Evidence"])
        with tab1:
            st.markdown(answer)
        with tab2:
            for doc in sources:
                with st.expander(f"Source: {doc.metadata['source']}"):
                    st.write(doc.page_content)
    else:
        st.warning("Database not found. Please click 'Refresh & Index' in the sidebar first.")