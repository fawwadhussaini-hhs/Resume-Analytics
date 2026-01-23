import os
import io
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader
from google.oauth2 import service_account 

# LangChain & Gemini Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
FOLDER_ID = st.secrets["FOLDER_ID"]
DB_PATH = "./chroma_db"



# New Method
def get_drive_service():
    # This grabs the dictionary you created in secrets.toml
    creds_info = st.secrets["gcp_service_account"]
    
    # Create credentials from the info dictionary
    creds = service_account.Credentials.from_service_account_info(
        creds_info, 
        scopes=SCOPES
    )
    
    return build('drive', 'v3', credentials=creds)


# --- 2. INCREMENTAL DATA INGESTION ---

def get_indexed_filenames():
    """Returns a set of filenames already present in the Chroma database."""
    if not os.path.exists(DB_PATH):
        return set()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # Get all metadata from the DB
    data = vector_db.get()
    if data and 'metadatas' in data:
        return {m['source'] for m in data['metadatas'] if 'source' in m}
    return set()


def process_files_to_chroma():
    service = get_drive_service()
    existing_files = get_indexed_filenames()

    # 1. Get ALL files from Drive using pagination
    all_drive_files = []
    page_token = None
    while True:
        query = f"'{FOLDER_ID}' in parents and mimeType = 'application/pdf' and trashed = false"
        results = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType)",
            pageSize=1000,
            pageToken=page_token
        ).execute()
        all_drive_files.extend(results.get('files', []))
        page_token = results.get('nextPageToken')
        if not page_token:
            break

    # 2. Filter for NEW files only
    new_files = [f for f in all_drive_files if f['name'] not in existing_files]

    if not new_files:
        st.info("No new resumes found. Everything is up to date!")
        return

    st.write(f"Found {len(new_files)} new resumes to index...")

    documents = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 3. Process new files in small batches to manage memory
    batch_size = 20
    for i, file in enumerate(new_files):
        try:
            status_text.text(f"Downloading ({i + 1}/{len(new_files)}): {file['name']}")

            request = service.files().get_media(fileId=file['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            fh.seek(0)
            reader = PdfReader(fh)
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": file['name']}))

            # Update progress
            progress_bar.progress((i + 1) / len(new_files))

            # Every 'batch_size' files, save to Chroma to prevent data loss
            if len(documents) >= batch_size:
                save_to_vector_db(documents)
                documents = []  # Clear memory

        except Exception as e:
            st.error(f"Error processing {file['name']}: {e}")

    # Final save for any remaining docs
    if documents:
        save_to_vector_db(documents)

    st.success(f"Successfully indexed {len(new_files)} new resumes!")


def save_to_vector_db(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )


def get_resume_count():
    """Returns (Total in Drive, Total Indexed in DB)"""
    try:
        service = get_drive_service()
        drive_total = 0
        page_token = None
        while True:
            query = f"'{FOLDER_ID}' in parents and mimeType = 'application/pdf' and trashed = false"
            res = service.files().list(q=query, fields="nextPageToken, files(id)", pageSize=1000,
                                       pageToken=page_token).execute()
            drive_total += len(res.get('files', []))
            page_token = res.get('nextPageToken')
            if not page_token: break

        indexed_total = len(get_indexed_filenames())
        return drive_total, indexed_total
    except:
        return 0, 0


# --- 3. RAG BRAIN ---

def get_llm_analysis(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Using Gemini 1.5 Flash (2.5 does not exist yet; 1.5 is the latest stable high-speed model)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # Increased K to 15 because you have 4,700+ records.
    # This gives the LLM a wider pool of candidates to compare.
    docs = vector_db.similarity_search(query, k=10)

    context = "\n\n".join([f"FILENAME: {d.metadata['source']}\nCONTENT: {d.page_content}" for d in docs])

    system_prompt = """
    You are an elite Talent Acquisition Specialist. Your goal is to find the absolute best candidates from the provided context.

    CONTEXT FROM RESUMES:
    {context}

    USER REQUIREMENT:
    {question}

    INSTRUCTIONS:
    - Compare the candidates against the requirements.
    - If multiple candidates are great, rank them.
    - Be specific about why their skills match.

    RESPONSE FORMAT:
    1. **Top Candidates Ranking**
    2. **Match Score (0-100)** per candidate.
    3. **Matching Skills & Experience**
    4. **Contact Info** (if available in text)
    """

    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})
    return response.content, docs


# --- 4. STREAMLIT UI ---

st.set_page_config(page_title="Resume AI Finder", layout="wide")
st.title("📂 Resume AI Analysis Engine (Enterprise Scale)")

drive_count, db_count = get_resume_count()

c1, c2, c3 = st.columns(3)
c1.metric("Resumes in Google Drive", drive_count)
c2.metric("Resumes Indexed in AI", db_count)
c3.metric("Pending Sync", drive_count - db_count)

with st.sidebar:
    st.header("Database Management")
    if st.button("🔄 Sync New Resumes Only", key="sync_btn"):
        process_files_to_chroma()
        st.rerun()

    st.divider()
    st.caption("This system only downloads resumes that haven't been added to the database yet.")

user_query = st.text_input("What kind of talent are you looking for?")

if user_query:
    if os.path.exists(DB_PATH) and db_count > 0:
        with st.spinner(f"Searching through {db_count} resumes..."):
            answer, sources = get_llm_analysis(user_query)

        tab1, tab2 = st.tabs(["📊 AI Recruitment Report", "🔍 Source Resume Snippets"])
        with tab1:
            st.markdown(answer)
        with tab2:
            # Show sources grouped by filename for cleaner UI
            for doc in sources:
                with st.expander(f"Extract from: {doc.metadata['source']}"):
                    st.write(doc.page_content)
    else:
        st.warning("The database is empty. Please use the sidebar to Sync Resumes.")