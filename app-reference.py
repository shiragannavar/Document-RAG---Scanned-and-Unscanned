import os
import time

import streamlit as st
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from astrapy import DataAPIClient
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from concurrent.futures import ThreadPoolExecutor

from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io


# Load environment variables
load_dotenv()
# st.set_page_config(initial_sidebar_state="collapsed")
fcategory=""
fname=""
LOCAL_SECRETS = False

if LOCAL_SECRETS:
    ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
else:
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

ASTRA_DB_KEYSPACE = "default_keyspace"
# ASTRA_DB_COLLECTION = "brooksfield"
ASTRA_DB_COLLECTION = "dcirrus"

client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
database = client.get_database(st.secrets["ASTRA_DB_ENDPOINT"])
collection = database.get_collection(ASTRA_DB_COLLECTION)

# Initialize session_state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_filename' not in st.session_state:
    st.session_state.selected_filename = "ALL"  # Default to "ALL"


def check_username():
    greeting_message = "Welcome to the document analysis assistant. How can I assist you?"
    username_prompt = "Please enter your username to continue:"

    if 'username_valid' not in st.session_state:
        st.session_state.username_valid = False
        st.write(greeting_message)

    username = st.text_input(username_prompt, key='username')

    if username:
        if not st.session_state.get('username_valid', False):
            st.session_state.username_valid = True
            st.session_state.user = username
    else:
        st.session_state.username_valid = False

    return st.session_state.username_valid


if not check_username():
    st.stop()

username = st.session_state.user
is_admin = username == "admin"


@st.cache_resource(show_spinner='Getting the Embedding Model...')
def load_embedding():
    return OpenAIEmbeddings()


@st.cache_resource(show_spinner='Getting the Vector Store from Astra DB...')
def load_vectorstore():
    return AstraDBVectorStore(
        embedding=load_embedding(),
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=st.secrets["ASTRA_DB_ENDPOINT"],
    )


@st.cache_resource(show_spinner='Getting the Chat Model...')
def load_model():
    # return OpenAI(openai_api_key=OPENAI_API_KEY)
    return ChatOpenAI(openai_api_key=OPENAI_API_KEY, max_tokens=4000, model_name='gpt-4o', streaming=True)

embedding = load_embedding()
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()
llm = load_model()

# Define the prompt template
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert legal assistant for question-answering tasks. Use only the following context to answer the question. If you don't know the answer, just say that you don't know. Be as verbose in your response as possible.

    context: {context}
    Question: "{question} and show page numbers as well as reference"
    Answer:
    """
)

CLASSIFY_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert document classifier. Your task is to classify the text in one of the below categories:
Legal
Financial
Corporate
Operational
Your reply should contain only the category. Do not explain anything.

    Text to Categorize: {classify_data}
    Answer:
    """
)

def get_text_for_category(text):
    words = text.split()
    first_1500_words = words[:1500]
    return ' '.join(first_1500_words)


def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    page_texts = []
    for page_num, page in enumerate(pdf.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text
            page_texts.append((page_text, page_num + 1))  # Store text with page number
    return text, page_texts

def is_scanned_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img)
        if text.strip():
            return True
    return False

def extract_text_from_scanned_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    page_texts = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_text = pytesseract.image_to_string(img)
        text += page_text
        page_texts.append((page_text, page_num + 1))  # Store text with page number
    return text, page_texts

def embed_and_store_text(text, page_texts, filename, owner, title, category):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = []
    for page_text, page_num in page_texts:
        texts = text_splitter.split_text(page_text)
        for chunk in texts:
            documents.append(Document(page_content=chunk, metadata={"filename": filename, "owner": owner, "title": title, "category": category, "page_number": page_num}))
    data_store = vectorstore.add_documents(documents)
    print(data_store)
    st.write(f"PDF File {filename} stored with {len(documents)} chunks.")


@st.cache_data(ttl=1)
def get_all_filenames(username):
    if username == "admin":
        # Admins have access to all files
        all_documents_cursor = collection.find({}, projection={"metadata.filename": True, "metadata.title": True, "metadata.category": True})
    else:
        # Regular users have access only to their files
        all_documents_cursor = collection.find({"metadata.owner": username}, projection={"metadata.filename": True, "metadata.title": True, "metadata.category": True})

    # Convert the cursor to a list to avoid CursorIsStartedException
    all_documents = list(all_documents_cursor)

    # Create a list to store the details of each document
    document_details = [
        {
            'filename': doc['metadata']['filename'],
            'title': doc['metadata']['title'],
            'category': doc['metadata']['category']
        }
        for doc in all_documents if 'metadata' in doc and 'filename' in doc['metadata']
    ]

    # Remove duplicates
    unique_documents = []
    seen = set()
    for doc in document_details:
        doc_tuple = (doc['filename'], doc['title'], doc['category'])
        if doc_tuple not in seen:
            seen.add(doc_tuple)
            unique_documents.append(doc)

    return unique_documents


def handle_chat(question, selected_filename):
    print(selected_filename)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        def get_filtered_documents(query):
            query_vector = embedding.embed_query(query)

            if selected_filename == "ALL" and is_admin:
                # Admin wants all documents, no filtering by owner
                cursor = collection.find(
                    sort={"$vector": query_vector},
                    projection={"content": True},
                )
            elif selected_filename == "ALL":
                cursor = collection.find(
                    {"metadata.owner": username},  # Filter documents by the current user
                    sort={"$vector": query_vector},
                    projection={"content": True},
                )
            else:
                cursor = collection.find(
                    {"metadata.filename": selected_filename} if is_admin else {"metadata.filename": selected_filename,
                                                                               "metadata.owner": username},
                    # Filter by filename and owner
                    sort={"$vector": query_vector},
                    projection={"*": True},
                )

            documents = list(cursor)  # Convert cursor to a list to use len() and iterate over it
            # st.write(f"Retrieved {len(documents)} documents for the query: {query}")
            result = []
            for doc in documents:
                if 'content' in doc:
                    text_data = f"content:{doc['content']}, page_number:{doc['metadata']['page_number']}"
                    result.append(text_data)
                    #print(result)
            return result

            # return [doc['content'] for doc in documents if 'content' in doc]

        context_documents = get_filtered_documents(question)

        # Limit to top 3 most relevant documents
        context_documents = context_documents[:4]

        if not context_documents:
            response_placeholder.markdown("I couldn't find any relevant content in the selected document.")
            st.write("No documents found matching the query.")
            return

        context = "\n\n".join(context_documents)
        # st.write(f"Context formed for LLM (first 500 chars): {context[:500]}...")

        # Build the chain with prompt and LLM
        chain = LLMChain(
            llm=llm,
            prompt=ANSWER_PROMPT
        )

        # Execute the chain
        ans = chain({"context": context, "question": question})

        # Ensure the response is a string
        response_content = str(ans["text"])

        # Display and store the response
        response_placeholder.markdown(response_content)
        st.session_state.messages.append(AIMessage(content=response_content))


if st.session_state.messages:
    for message in st.session_state.messages:
        st.chat_message(message.type).markdown(message.content)


# def refresh_filenames():
#     filenames = get_all_filenames(username)
#     if uploaded_file and uploaded_file.name not in filenames:
#         filenames.append(uploaded_file.name)
#     if "ALL" not in filenames:
#         filenames.insert(0, "ALL")
#     st.session_state['filenames'] = filenames

def refresh_filenames(file_name, title_name, category_name):
    # Fetch the latest filenames with their metadata
    filenames = get_all_filenames(username)

    # Check if an uploaded file exists and is not already in the list
    if uploaded_file:
        uploaded_file_name = uploaded_file.name
        if not any(doc['filename'] == uploaded_file_name for doc in filenames):
            # Add the uploaded file's metadata (assuming you have title and category for the uploaded file)
            filenames.append({
                'filename': file_name,
                'title': title_name,
                'category': category_name
            })

    # Store the filenames in session state
    st.session_state['filenames'] = filenames


with st.sidebar:
    st.image("https://i0.wp.com/opensource.org/wp-content/uploads/2023/01/datastax-logo-square_transparent-background.png", use_column_width=True)
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)

    if uploaded_file is not None:
        title = st.text_input("Enter the title of the PDF document:")
        upload_button = st.button("Upload")
        with st.spinner('Processing uploaded file...'):
            if upload_button:
                if not title:
                    st.warning("Please enter the title of the PDF document before uploading.")
                else:
                    # os.makedirs("PDF files", exist_ok=True)
                    # # Save the uploaded file to the directory
                    # file_path = os.path.join("PDF files", uploaded_file.name)
                    # with open(file_path, "wb") as f:
                    #     f.write(uploaded_file.getbuffer())
                    # st.success(f"File saved to {file_path}")

                    file_bytes = uploaded_file.read()
                    file_stream = io.BytesIO(file_bytes)

                    with ThreadPoolExecutor() as executor:
                        if is_scanned_pdf(file_stream):
                            file_stream.seek(0)  # Reset file pointer before passing to the function
                            future = executor.submit(extract_text_from_scanned_pdf, file_stream)
                        else:
                            file_stream.seek(0)  # Reset file pointer before passing to the function
                            future = executor.submit(extract_text_from_pdf, file_stream)
                        text, page_texts = future.result()

                    # with ThreadPoolExecutor() as executor:
                    #     future = executor.submit(extract_text_from_pdf, uploaded_file)
                    #     text = future.result()
                    # Build the chain with prompt and LLM
                    chain_category = LLMChain(
                        llm=llm,
                        prompt=CLASSIFY_PROMPT,
                    )
                    data_to_analyze = get_text_for_category(text)
                    # Execute the chain
                    ans_classify = chain_category({"classify_data": data_to_analyze})
                    print(ans_classify['text'])
                    st.info(f'{ans_classify['text']}', icon="ℹ️")
                    st.toast(f'Category is: {ans_classify['text']}', icon="⚡")
                    embed_and_store_text(text,page_texts, uploaded_file.name, username, title, ans_classify['text'])
                    st.success("PDF content has been processed and stored successfully. ⚡")
                    refresh_filenames(uploaded_file.name, title, ans_classify['text'])  # Refresh filenames after upload

    # Initialize session state for filenames
    if 'filenames' not in st.session_state:
        st.session_state['filenames'] = get_all_filenames(username)

    # Extract titles for the dropdown
    # titles = ["ALL"] + [doc['title'] for doc in st.session_state['filenames']]
    titles = ["ALL"] + [doc["title"] for doc in st.session_state["filenames"]]
    print(titles)
    # Make sure selected_filename persists
    selected_title = st.selectbox(
        "Select a document you uploaded for running queries:",
        titles,
        index=titles.index(st.session_state.get('selected_title', "ALL")),
        key='selected_title'
    )

    # Find the selected document details
    if selected_title != "ALL":
        selected_doc = next((doc for doc in st.session_state['filenames'] if doc['title'] == selected_title), None)
        if selected_doc:
            fname = selected_doc['filename']
            fcategory = selected_doc['category']
            st.write(f"Filename: {selected_doc['filename']}")
            st.write(f"Category: {selected_doc['category']}")
    else:
        st.write("No document selected.")
    text=""
    if fcategory.strip() == "Operational":
        text = """
        - <b>Process-related:</b> "Can you explain the steps involved in [process name]?"</br>
        - <b>Resource allocation:</b> "How are resources allocated for [task or project]?"</br>
        - <b>Performance metrics:</b> "What are the key performance indicators for [department or function]?"</br>
        - <b>SOP compliance:</b> "Are there any specific Standard Operating Procedures (SOPs) for [task or process]?"</br>
        - <b>Risk management:</b> "How are risks mitigated in [project or operation]?"
        """
    elif fcategory.strip() == "Legal":
        text = """
        - <b>Contractual Obligations:</b> "What are the specific obligations of [party name] under this contract?"</br>
        - <b>Dispute Resolution:</b> "How are disputes resolved between the parties?"</br>
        - <b>Intellectual Property Rights:</b> "What intellectual property rights are granted or licensed under this agreement?"</br>
        - <b>Governing Law and Jurisdiction:</b> "What law governs this contract, and where are disputes to be resolved?"</br>
        - <b>Confidentiality and Non-Disclosure:</b> "What information is considered confidential, and what are the restrictions on its disclosure?"
        """
    elif fcategory.strip() == "Financial":
        text = """
        - <b>Financial Statements:</b> "Can you provide a breakdown of the company's income statement, balance sheet, and cash flow statement?"</br>
        - <b>Financial Ratios:</b> "What are the key financial ratios that measure the company's profitability, liquidity, and solvency?"</br>
        - <b>Auditing:</b> "Who is the company's auditor, and what are the findings of the most recent audit?"</br>
        - <b>Tax Liabilities:</b> "What are the company's tax liabilities and obligations?"</br>
        - <b>Investment Strategies:</b> "What is the company's investment strategy, and where does it allocate its capital?"
        """
    elif fcategory.strip() == "Corporate":
        text = """
        - <b>Company Structure:</b> "How is the company organized (e.g., sole proprietorship, partnership, corporation)?"</br>
        - <b>Board of Directors:</b> "Who are the members of the board of directors, and what are their responsibilities?"</br>
        - <b>Shareholder Rights:</b> "What are the rights of shareholders, including voting rights and dividend distribution?"</br>
        - <b>Corporate Governance Policies:</b> "What are the company's policies on corporate governance, such as conflict of interest and ethical conduct?"</br>
        - <b>Risk Management:</b> "How does the company identify and mitigate risks?"
        """

    st.markdown("<b>Questions you can ask</b>", unsafe_allow_html=True)
    st.markdown(f"<small>{text}</small>", unsafe_allow_html=True)
# Handle chat input and processing
if question := st.chat_input("How can I help you?", key='chat_input'):
    st.session_state.messages.append(HumanMessage(content=question))
    handle_chat(question, selected_doc['filename'])

if "initial_greeted" not in st.session_state:
    st.session_state.initial_greeted = True

    for message in st.session_state.messages:
        st.chat_message(message.type).markdown(message.content, unsafe_allow_html=True)