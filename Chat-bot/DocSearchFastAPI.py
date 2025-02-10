import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import CharacterTextSplitter
import shutil
import uvicorn
import logging
import chardet
from markitdown import MarkItDown
import tempfile
import hashlib

# Load environment variables from .env file
dotenv_path = find_dotenv()
if not dotenv_path:
    raise FileNotFoundError("Could not find .env file")
load_dotenv(dotenv_path)

# Retrieve variables from the environment
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Provide a default model if not set

# Retrieve and validate user IDs and groups
valid_user_ids_str = os.getenv("VALID_USER_IDS", "")
user_groups_str = os.getenv("USER_GROUPS", "")

try:
    VALID_USER_IDS = {int(user_id) for user_id in valid_user_ids_str.split(",") if user_id.strip()}
    USER_GROUPS = {}
    for user_group in user_groups_str.split(";"):
        if ":" in user_group:
            user_id, groups = user_group.split(":")
            USER_GROUPS[int(user_id)] = {int(group) for group in groups.split(",")}
        else:
            raise ValueError
except ValueError:
    raise ValueError("VALID_USER_IDS or USER_GROUPS environment variable contains invalid values")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the uploaded_files directory exists
os.makedirs("uploaded_files", exist_ok=True)

# Initialize LangChain components
embeddings = OpenAIEmbeddings(api_key=API_KEY)
vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_db")

# Define a prompt template for similarity search
similarity_prompt = PromptTemplate(
    input_variables=["input", "docs"],
    template="Use the following documents to answer the question: {docs}\nQuestion: {input}\nAnswer:"
)

# Convert files to markdown format 
def convert_to_markdown(file_path: str) -> str:
    logging.info(f"Converting file to markdown: {file_path}")
    markitdown = MarkItDown()
    """Convert document to markdown format"""
    try:
        result = markitdown.convert(file_path)
        logging.info("Conversion to markdown successful")
        markdown_content = result.text_content

        # Generate markdown stamp
        markdown_stamp = generate_markdown_stamp(markdown_content)
        logging.info(f"Generated markdown stamp: {markdown_stamp}")

        return markdown_content, markdown_stamp
    except Exception as e:
        logging.error(f"Error converting document to markdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_markdown_stamp(content: str) -> str:
    """Generate a unique stamp for the markdown content"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

# Initialize the LLM
llm = ChatOpenAI(api_key=API_KEY, model=MODEL)
llm_chain = LLMChain(llm=llm, prompt=similarity_prompt)

def get_current_user(user_id: str = Header(...)):
    try:
        user_id = int(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="User ID must be an integer")
    
    if user_id not in VALID_USER_IDS:
        raise HTTPException(status_code=401, detail="Invalid user ID")
    return user_id

def get_user_groups(user_id: int):
    return USER_GROUPS.get(user_id, set())

@app.get("/")
def read_root():
    logging.info("Endpoint '/' called")
    return {"message": "Welcome to my app!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: int = Depends(get_current_user), group_id: int = Form(...)):
    if group_id not in get_user_groups(user_id):
        raise HTTPException(status_code=403, detail="You do not have permission to upload to this group")
    
    logging.info(f"Uploading file: {file.filename}")

    try:
        file_location = f"uploaded_files/{file.filename}"
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)    
        logging.info(f"Saving file to {file_location}")

        content, markdown_stamp = convert_to_markdown(file_location)
        logging.info(f"Content of the file: {content}")

        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            # Write the in-memory data to the temporary file
            temp_file.write(content)
            # Get the path of the temporary file
            temp_file_path = temp_file.name
            logging.info(f"Temporary file path: {temp_file_path}")


        loader = UnstructuredMarkdownLoader(temp_file_path)
        document=loader.load()
        logging.info(f"Content of the file: {content}")

        # Clean up: remove the temporary file after processing
        os.remove(temp_file_path)

        #split the  document into sentences
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(document)

        # Embed metadata into each document

        # convert the groups to string in JSON format
        logging.info(f"Access control groups: {group_id}")
        metadata = {
            "access_control_groups": group_id,
            "author": "John Doe",
            "timestamp": "2023-10-01T12:00:00Z",
            "filename": file_location,
            "markdown_stamp": markdown_stamp  # Add markdown stamp to metadata
        }
        for doc in texts:
            doc.metadata.update(metadata)
        
        # Filter complex metadata
        texts = filter_complex_metadata(texts)

        try:
            logging.info("Using Chroma vector store")
            #vectorstore.add_texts([content], metadata)
            vectorstore.add_documents(texts)  # Removed metadata argument

        except Exception as e:
            logging.error(f"Error adding texts to vectorstore: {e}")
            return JSONResponse(
                content={"error": f"Error adding texts to vectorstore: {e}"}, 
                status_code=500
            )

        return JSONResponse(content={"message": "File uploaded successfully"}, status_code=200)
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/search")
async def search_docs(query: str = Form(...), user_id: int = Depends(get_current_user)):
    # Get user groups
    user_groups = get_user_groups(user_id)
    logging.info(f"Searching documents with query: '{query}' using groups: {user_groups}")
    try:
        # Perform similarity search on filtered documents based on user groups
        unique_docs = {}
        for group in user_groups:
            # Limit the number of documents to 20 per search
            docs = vectorstore.similarity_search(query, filter={"access_control_groups": group}, k=20)
            # logging.info(f"Found these docs: '{docs}'") # For debugging
            # Find unique documents based on markdown_stamp
            for doc in docs:
                stamp = doc.metadata.get("markdown_stamp")
                # logging.info(f"Found stamp: '{stamp}'") # For debugging
                if stamp and stamp not in unique_docs:
                    unique_docs[stamp] = doc
        # Convert to JSON serializable format
        docs_json = [{"content": d.page_content, "metadata": d.metadata} for d in unique_docs.values()]
        # Run the LLM chain
        response = llm_chain.invoke({"input": query, "docs": docs_json})
        logging.info("Search successful, returning response")
        return JSONResponse(content={"response": response}, status_code=200)
    
    except Exception as e:
        logging.error(f"Error searching docs: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
