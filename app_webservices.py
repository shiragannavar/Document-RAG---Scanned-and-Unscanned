from flask import Flask, jsonify, request
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
from urllib.request import urlopen
from io import BytesIO

from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

load_dotenv()
LOCAL_SECRETS = False

if LOCAL_SECRETS:
    ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
else:
    ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

ASTRA_DB_KEYSPACE = "default_keyspace"
ASTRA_DB_COLLECTION = "dcirrus"

client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
database = client.get_database(os.environ["ASTRA_DB_ENDPOINT"])
collection = database.get_collection(ASTRA_DB_COLLECTION)

embedding = OpenAIEmbeddings()
vectorstore = AstraDBVectorStore(
    embedding=embedding,
    collection_name=ASTRA_DB_COLLECTION,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=os.environ["ASTRA_DB_ENDPOINT"],
)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, max_tokens=4000, model_name='gpt-4o', streaming=True)

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
    return f"PDF File {filename} stored with {len(documents)} chunks."

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

app = Flask(__name__)

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    file = request.files['file']
    title = request.form['title']
    username = request.form['username']
    if is_scanned_pdf(file):
        file.seek(0)  # Reset file pointer before passing to the function
        text, page_texts = extract_text_from_scanned_pdf(file)
    else:
        file.seek(0)  # Reset file pointer before passing to the function
        text, page_texts = extract_text_from_pdf(file)
    chain_category = LLMChain(
        llm=llm,
        prompt=CLASSIFY_PROMPT,
    )
    data_to_analyze = get_text_for_category(text)
    ans_classify = chain_category({"classify_data": data_to_analyze})
    category = ans_classify['text']
    embed_and_store_text(text, page_texts, file.filename, username, title, category)
    return jsonify({'message': f'PDF File {file.filename} stored with {len(page_texts)} chunks.'})

@app.route('/upload_pdf_url', methods=['POST'])
def upload_pdf_url():
    url = request.json['url']
    filename = request.json['filename']
    title = request.json['title']
    username = request.json['username']

    try:
        response = urlopen(url)
        file = BytesIO(response.read())
        if is_scanned_pdf(file):
            file.seek(0)  # Reset file pointer before passing to the function
            text, page_texts = extract_text_from_scanned_pdf(file)
        else:
            file.seek(0)  # Reset file pointer before passing to the function
            text, page_texts = extract_text_from_pdf(file)
        chain_category = LLMChain(
            llm=llm,
            prompt=CLASSIFY_PROMPT,
        )
        data_to_analyze = get_text_for_category(text)
        ans_classify = chain_category({"classify_data": data_to_analyze})
        category = ans_classify['text']
        embed_and_store_text(text, page_texts, filename, username, title, category)
        return jsonify({'message': f'PDF File {filename} stored with {len(page_texts)} chunks.'})
    except Exception as e:
        return jsonify({'error': str(e)})





@app.route('/get_filenames', methods=['GET'])
def get_filenames():
    username = request.args.get('username')
    filenames = get_all_filenames(username)
    return jsonify(filenames)

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json['question']
    filename = request.json['filename']
    username = request.json['username']
    context_documents = []
    if filename == "ALL":
        cursor = collection.find(
            {"metadata.owner": username},
            projection={"content": True},
        )
    else:
        cursor = collection.find(
            {"metadata.filename": filename, "metadata.owner": username},
            projection={"content": True},
        )
    documents = list(cursor)  # Convert cursor to a list to use len() and iterate over it

    for doc in documents:
        if 'content' in doc and 'metadata' in doc and 'page_number' in doc['metadata']:
            text_data = f"content:{doc['content']}, page_number:{doc['metadata']['page_number']}"
            context_documents.append(text_data)
        elif 'content' in doc:
            text_data = f"content:{doc['content']}"
            context_documents.append(text_data)
            
    context = "\n\n".join(context_documents)
    chain = LLMChain(
        llm=llm,
        prompt=ANSWER_PROMPT
    )
    ans = chain({"context": context, "question": question})
    return jsonify({'answer': ans['text']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)