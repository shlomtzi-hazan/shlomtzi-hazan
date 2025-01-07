import base64
from dotenv import load_dotenv, find_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.tools import Tool 
import smtplib
from email.mime.text import MIMEText
import requests
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from google_auth_oauthlib.flow import InstalledAppFlow
import json

# Load environment variables from .env file
dotenv_path = find_dotenv()
if not dotenv_path:
    raise FileNotFoundError("Could not find .env file")
load_dotenv(dotenv_path)

# Retrieve variables from the environment
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

SEARX_HOST = os.getenv("SEARX_HOST")
SEARX_ENGINES = os.getenv("SEARX_ENGINES")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Provide a default model if not set
SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = os.getenv('SMTP_PORT')
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')

# Retrieve Gmail API credentials from the environment
GMAIL_CLIENT_SECRETS = os.getenv('GMAIL_CLIENT_SECRETS')

# Load the Gmail client secrets from the file
try:
    with open(GMAIL_CLIENT_SECRETS, 'r') as file:
        gmail_client_secrets_dict = json.load(file)
    print(gmail_client_secrets_dict)
except (TypeError, json.JSONDecodeError) as e:
    print(f"Error loading Gmail client secrets: {e}")
    gmail_client_secrets_dict = None

# Define the scopes for Gmail, Drive, and Documents
SCOPES = [
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents.readonly"
]

# Initialize credentials with InstalledAppFlow

try:
    # Use InstalledAppFlow for user authorization
    flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CLIENT_SECRETS, SCOPES)
    credentials = flow.run_local_server(port=0)
    print("Successfully authenticated and acquired credentials.")

    # Save credentials for later use (optional)
    with open("token.json", "w") as token_file:
        token_file.write(credentials.to_json())
except Exception as e:
    print(f"Error during authentication: {e}")

# Initialize Gmail and Drive services with the acquired credentials
try:
    api_resource = build('gmail', 'v1', credentials=credentials)
    drive_service = build('drive', 'v3', credentials=credentials)
    print("Gmail and Drive services initialized successfully.")
except Exception as e:
    print(f"Error initializing services: {e}")

# Initialize Google Drive service
drive_service = build('drive', 'v3', credentials=credentials)

# Define a prompt template for RAG
rag_prompt = PromptTemplate(
    input_variables=["input", "docs"],
    template="Use the following documents to answer the question: {docs}\nQuestion: {input}\nAnswer:"
)

# Initialize the LLM
llm = ChatOpenAI(api_key=API_KEY, model=MODEL)
llm_chain = LLMChain(llm=llm, prompt=rag_prompt)

# Define the tools
def conversetional_input(user_input):
    print(f"Searching for: {user_input}")
    return llm(user_input)

def email_input(to_email, subject, message):
    try:
        message = MIMEText(message)
        message['to'] = to_email
        message['subject'] = subject
        raw_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

        send_message = (api_resource.users().messages().send(userId="me", body=raw_message).execute())
        print(f"Message Id: {send_message['id']}")
        print("Email sent successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

def searx_search(query):
    params = {
        'q': query,
        'engines': SEARX_ENGINES,
        'format': 'json'
    }
    response = requests.get(f"{SEARX_HOST}/search", params=params)
    results = response.json()
    return results

def list_files(service, query):
    try:
        results = service.files().list(q=query, pageSize=10, fields="files(id, name)").execute()
        items = results.get('files', [])
        return items
    except HttpError as error:
        print(f"An error occurred: {error}")
        return []

def download_file(service, file_id):
    try:
        request = service.files().get_media(fileId=file_id)
        file_content = request.execute()
        return file_content.decode('utf-8')
    except HttpError as error:
        print(f"An error occurred: {error}")
        return ""

def retrieve_docs(query):
    files = list_files(drive_service, query)
    docs = []
    for file in files:
        content = download_file(drive_service, file['id'])
        docs.append(content)
    return docs

tools = [
    Tool(
        name="Conversational Input",
        func=conversetional_input,
        description="This tool is used to provide a conversational response based on the LLM. It is invoked when the user query doesn't require an online search (e.g., general knowledge, basic information, or pre-existing knowledge within the model)."
    ),
    Tool(
        name="email_input",
        func=email_input,
        description="Send an email with the provided subject and message using Gmail API"
    ),
    Tool(
        name="searx-search",
        func=searx_search,
        description="This tool is used to perform an online search using the Searx search engine."
    ),
    Tool(
        name="retrieve_docs",
        func=retrieve_docs,
        description="This tool is used to retrieve documents from Google Drive based on the query."
    )
]

# Initialize the agent with the tools
agent = initialize_agent(
    tools=tools,
    llm=llm_chain,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

def chat():
    print("Chat started! Type 'exit' to end the conversation.\nIf you'd like to send the data provided to your email, please type 'send email'.\nIf you'd like to find information within your Google Docs or Drive, please type 'search docs'.")
    last_response = ""

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Ending chat.")
            break

        # Send email functionality
        elif "send email" in user_input.lower():
            print("Please provide the email address you'd like to send the data to: ")
            to_email = input("Email: ").strip()
            print("Please provide the subject of the email: ")
            subject = input("Subject: ").strip()

            # Attempt to send the email
            try:
                email_input(to_email=to_email, subject=subject, message=last_response)
            except Exception as e:
                print(f"Failed to send email: {e}")
            continue

        # Search docs functionality
        elif "search docs" in user_input.lower():
            try:
                query = user_input.replace("search docs", "").strip()
                docs = retrieve_docs(query)
                if not docs:
                    response = "I couldn't find any relevant documents. Can you provide more details or try a different query?"
                else:
                    response = agent.invoke({"input": query, "docs": docs})
                print(f"OpenAI: {response}")
            except Exception as e:
                print(f"Failed to search docs: {e}")
            continue
        
        # Process input using the agent
        try:
            # For regular conversational input, provide an empty list for docs
            response = agent.run(user_input)  # Use the `run` method for user input
            if isinstance(response, dict) and 'output' in response:
                clean_response = response['output'].replace("\n", " ").strip()
            else:
                clean_response = response.replace("\n", " ").strip()

            # Store the last response and display it
            last_response = clean_response
            print(f"OpenAI: {clean_response}")
        except Exception as e:
            print(f"Error during chat processing: {e}") 

if __name__ == "__main__":
    chat()
