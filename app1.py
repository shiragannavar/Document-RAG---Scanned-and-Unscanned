import os
import streamlit as st
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from astrapy import DataAPIClient
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
st.set_page_config(initial_sidebar_state="collapsed")

LOCAL_SECRETS = False
if LOCAL_SECRETS:
    ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
else:
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

ASTRA_DB_KEYSPACE = "default_keyspace"
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
    return OpenAI(openai_api_key=OPENAI_API_KEY)

embedding = load_embedding()
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()
llm = load_model()

# Define the prompt template
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be as verbose and educational in your response as possible. Reply in as less words as possible
    context: {context}
    Question: "{question}"
    Answer: """
)

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def embed_and_store_text(text, filename, owner, title):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"filename": filename, "owner": owner, "title": title}) for chunk in texts]
    data_store = vectorstore.add_documents(documents)
    print(data_store)
    st.write(f"PDF File {filename} stored with {len(documents)} chunks.")

@st.cache_data(ttl=600)
def get_all_filenames(username):
    # Fetch documents based on the user's access level
    if username == "admin":
        # Admins have access to all files
        all_documents = collection.find({}, projection={"metadata.filename": True})
    else:
        # Regular users have access only to their files
        all_documents = collection.find({"metadata.owner": username}, projection={"metadata.filename": True})
    filenames = list(set(
        doc['metadata']['filename'] for doc in all_documents if 'metadata' in doc and 'filename' in doc['metadata']
    ))
    st.write(f"Available filenames: {filenames}")
    return filenames

def handle_chat(question, selected_filename):
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
                    {"metadata.filename": selected_filename} if is_admin else {"metadata.filename": selected_filename, "metadata.owner": username},  # Filter by filename and owner
                    sort={"$vector": query_vector},
                    projection={"content": True},
                )
            documents = list(cursor)  # Convert cursor to a list to use len() and iterate over it
            # st.write(f"Retrieved {len(documents)} documents for the query: {query}")
            return [doc['content'] for doc in documents if 'content' in doc]

        context_documents = get_filtered_documents(question)
        # Limit to top 3 most relevant documents
        context_documents = context_documents[:3]
        if not context_documents:
            response_placeholder.markdown("I couldn't find any relevant content in the selected document.")
            st.write("No documents found matching the query.")
            return

        context = "\n\n".join(context_documents)
        # st.write(f"Context formed for LLM (first 500 chars): {context[:500]}...")

        # Build the chain with prompt and LLM
        chain = LLMChain(
            llm=llm,
            prompt=ANSWER_PROMPT,
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

def refresh_filenames():
    filenames = get_all_filenames(username)
    if uploaded_file and uploaded_file.name not in filenames:
        filenames.append(uploaded_file.name)
    if "ALL" not in filenames:
        filenames.insert(0, "ALL")
    st.session_state['filenames'] = filenames

with st.sidebar:
    st.image("https://i0.wp.com/opensource.org/wp-content/uploads/2023/01/datastax-logo-square_transparent-background.png", use_column_width=True)
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        title = st.text_input("Enter the title of the PDF document:")
        upload_button = st.button("Upload")
        if upload_button:
            if not title:
                st.warning("Please enter the title of the PDF document before uploading.")
            else:
                # Create the directory if it doesn't exist
                os.makedirs("PDF files", exist_ok=True)
                # Save the uploaded file to the directory
                file_path = os.path.join("PDF files", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"File saved to {file_path}")

                with ThreadPoolExecutor() as executor:
                    future = executor.submit(extract_text_from_pdf, file_path)
                    text = future.result()
                embed_and_store_text(text, uploaded_file.name, username, title)
                st.success("PDF content has been embedded and stored successfully.")
                refresh_filenames()  # Refresh filenames after upload

if 'filenames' not in st.session_state:
    st.session_state['filenames'] = get_all_filenames(username)
    if "ALL" not in st.session_state['filenames']:
        st.session_state['filenames'].insert(0, "ALL")
filenames = st.session_state['filenames']

# Make sure selected_filename persists
selected_filename = st.selectbox("Select a document for search:", filenames, index=filenames.index(st.session_state.get('selected_filename', "ALL")), key='selected_filename')
print(selected_filename)

# Handle chat input and processing
if question := st.chat_input("How can I help you?", key='chat_input'):
    st.session_state.messages.append(HumanMessage(content=question))
    handle_chat(question, selected_filename)

if "initial_greeted" not in st.session_state:
    st.session_state.initial_greeted = True
for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content, unsafe_allow_html=True)