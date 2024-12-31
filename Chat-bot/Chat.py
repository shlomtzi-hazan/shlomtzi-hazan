import base64
from dotenv import load_dotenv
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
import json

# Load environment variables from .env file
load_dotenv()

# Retrieve variables from the environment
API_KEY = os.getenv("OPENAI_API_KEY")
SEARX_HOST = os.getenv("SEARX_HOST")
SEARX_ENGINES = os.getenv("SEARX_ENGINES").split(",")
MODEL = os.getenv("OPENAI_MODEL")
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

# Initialize Gmail API credentials and resource service
credentials = get_gmail_credentials(
    scopes=["https://mail.google.com/"],
    client_secrets_file=GMAIL_CLIENT_SECRETS,
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

#TODO: Creat many tools
#TODO: I want you to research bears and send the results to my email

def conversetional_input(user_input):
    print(f"Searching for: {user_input}")
    return llm(user_input)  # Corrected method to invoke the LLM

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

# Define the tools
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
    )
]

# Initialize the agent with the tools
llm = ChatOpenAI(api_key=API_KEY, model=MODEL)
agent = initialize_agent(
    tools=tools,
    llm=llm,  # Pass the plain LLM instance here, not the LLMChain
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

def chat():
    print("Chat started! Type 'exit' to end the conversation.\nIf you'd like to send the data provided to your email, type 'send email'.")
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

        # Process input using the agent
        try:
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
