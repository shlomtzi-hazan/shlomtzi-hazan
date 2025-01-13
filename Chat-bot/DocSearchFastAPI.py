import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import shutil
import uvicorn
import logging
import chardet

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

# Initialize the LLM
llm = ChatOpenAI(api_key=API_KEY, model=MODEL)
llm_chain = LLMChain(llm=llm, prompt=similarity_prompt)

@app.get("/")
def read_root():
    return {"message": "Welcome to my app!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_location = f"uploaded_files/{file.filename}"
        logging.info(f"Saving file to {file_location}")

        # Save the uploaded file
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detect file encoding
        with open(file_location, "rb") as f:
            raw_data = f.read()
            detected_encoding = chardet.detect(raw_data)["encoding"]

        # Read file with detected encoding
        encoding_to_use = detected_encoding if detected_encoding else "utf-8"
        with open(file_location, "r", encoding=encoding_to_use) as f:
            content = f.read()
            try:
                vectorstore.add_texts([content], [{"filename": file.filename}])
            except Exception as e:
                logging.error(f"Error adding texts to vectorstore: {e}")
                return JSONResponse(content={"error": f"Error adding texts to vectorstore: {e}"}, status_code=500)

        return JSONResponse(content={"message": "File uploaded successfully"}, status_code=200)
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/search")
async def search_docs(query: str = Form(...)):
    try:
        # Perform similarity search in Chroma vectorstore
        docs = vectorstore.similarity_search(query)
        if not docs:
            return JSONResponse(content={"message": "No relevant documents found"}, status_code=404)
        
        # Convert documents to JSON-serializable format
        docs_json = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        
        # Generate response using LLM
        response = llm_chain.invoke({"input": query, "docs": docs_json})
        return JSONResponse(content={"response": response}, status_code=200)
    except Exception as e:
        logging.error(f"Error searching docs: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)
