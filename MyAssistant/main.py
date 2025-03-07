import os
import json
import logging
import argparse
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv
from jinja2 import Template
from datetime import datetime
import sys

from chat_types import Role, Message, Conversation, TechGuidelineInfo
import PyPDF2

# Load environment variables
load_dotenv()

# Configure OpenAI API
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4-turbo')
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '2000'))

# Set up logging
if os.getenv('ENABLE_LOGGING', 'False').lower() == 'true':
    log_path = os.getenv('LOG_PATH', 'logs/chat_logs.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
else:
    logging.basicConfig(level=logging.WARNING)


class OakieAssistant:
    def __init__(self, model_path: str = 'MyAssistant/model.json', prompt_template_path: str = 'MyAssistant/prompt.jinja2'):
        """Initialize the Oakie AI assistant.
        
        Args:
            model_path: Path to the JSON file containing company guidelines
            prompt_template_path: Path to the Jinja2 template for system prompt
        """
        self.guidelines = self._load_guidelines(model_path)
        self.system_prompt = self._create_system_prompt(prompt_template_path)
        self.conversation = self._initialize_conversation()
        self.documents = {}  # Dictionary to store document contents by filename
        
    def _load_guidelines(self, model_path: str) -> TechGuidelineInfo:
        """Load company guidelines from JSON file."""
        try:
            with open(model_path, 'r') as file:
                guidelines_data = json.load(file)
            return TechGuidelineInfo.from_dict(guidelines_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load guidelines: {e}")
            raise
    
    def _create_system_prompt(self, template_path: str) -> str:
        """Create system prompt from template and guidelines."""
        try:
            with open(template_path, 'r') as file:
                template_content = file.read()
            
            template = Template(template_content)
            return template.render(
                company_name=self.guidelines.company_name,
                tech_stack=self.guidelines.tech_stack,
                coding_conventions=self.guidelines.coding_conventions,
                best_practices=self.guidelines.best_practices
            )
        except FileNotFoundError as e:
            logging.error(f"Failed to load prompt template: {e}")
            raise
    
    def _initialize_conversation(self) -> Conversation:
        """Initialize a new conversation with the system prompt."""
        conversation = Conversation(messages=[])
        conversation.add_message(Role.SYSTEM, self.system_prompt)
        return conversation
    
    def add_document(self, filename: str, content: str) -> None:
        """Add a document to the assistant's context.
        
        Args:
            filename: Name of the document
            content: Content of the document
        """
        self.documents[filename] = content
        logging.info(f"Added document: {filename}")
    
    def load_document_from_file(self, filepath: str) -> bool:
        """Load a document from a file.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            if filepath.lower().endswith('.pdf'):
                with open(filepath, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    content = ""
                    for page_num in range(len(reader.pages)):
                        content += reader.pages[page_num].extract_text()
                    
                    filename = os.path.basename(filepath)
                    self.add_document(filename, content)
                    return True
            else:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                filename = os.path.basename(filepath)
                self.add_document(filename, content)
                return True
        except Exception as e:
            print(f"Error loading document from {filepath}: {e}")
            logging.error(f"Failed to load document from {filepath}: {e}")
            return False
            
    def format_documents_context(self) -> str:
        """Format loaded documents as context for the AI.
        
        Returns:
            Formatted document context as a string
        """
        if not self.documents:
            return ""
            
        context = "Here are some documents that might be relevant to the question:\n\n"
        for filename, content in self.documents.items():
            context += f"--- Document: {filename} ---\n{content}\n\n"
        return context
    
    def add_previous_mentions(self, mentions: List[Dict[str, str]]) -> None:
        """Add previous mentions to the conversation.
        
        Args:
            mentions: List of previous mentions as {role, content} dictionaries
        """
        if not mentions:
            return
            
        # Add a separator to mark these as explicit mentions
        self.conversation.add_message(Role.SYSTEM, "The user has explicitly mentioned the following messages:")
        
        # Add each mention
        for mention in mentions:
            role = Role(mention.get("role", "user"))
            content = mention.get("content", "")
            self.conversation.add_message(role, content)
            
        # Add a separator to mark the end of mentions
        self.conversation.add_message(Role.SYSTEM, "End of explicitly mentioned messages. Please consider them in your response.")
    
    def ask(self, user_query: str, include_documents: bool = True, 
            mentions: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a user query and get a response from the AI.
        
        Args:
            user_query: The user's question or request
            include_documents: Whether to include document context
            mentions: List of previous mentions to include
            
        Returns:
            The AI assistant's response
        """
        # Include document context if requested
        if include_documents and self.documents:
            doc_context = self.format_documents_context()
            if doc_context:
                self.conversation.add_message(Role.SYSTEM, doc_context)
        
        # Include previous mentions if provided
        if mentions:
            self.add_previous_mentions(mentions)
        
        # Add user message to conversation
        self.conversation.add_message(Role.USER, user_query)
        
        # Log the query
        logging.info(f"User query: {user_query}")
        
        try:
            # Call OpenAI API with the new client format
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.conversation.get_messages(),
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            # Extract assistant response - updated for new API format
            assistant_response = response.choices[0].message.content
            
            # Add assistant response to conversation history
            self.conversation.add_message(Role.ASSISTANT, assistant_response)
            
            # Log the response
            logging.info(f"Assistant response: {assistant_response[:100]}...")
            
            return assistant_response
            
        except Exception as e:
            error_msg = f"Error getting response from OpenAI: {str(e)}"
            logging.error(error_msg)
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def reset_conversation(self, keep_documents: bool = True) -> None:
        """Reset the conversation to start fresh.
        
        Args:
            keep_documents: Whether to keep loaded documents
        """
        self.conversation = self._initialize_conversation()
        if not keep_documents:
            self.documents = {}
        logging.info("Conversation reset")


def process_command_line_args():
    """Process command line arguments."""
    parser = argparse.ArgumentParser(description="Oakie Technologies AI Assistant")
    parser.add_argument("--docs", "-d", nargs="+", help="Paths to documents to load for context")
    parser.add_argument("--query", "-q", help="Query to ask (if not in interactive mode)")
    parser.add_argument("--non-interactive", "-n", action="store_true", help="Run in non-interactive mode")
    return parser.parse_args()


def main():
    """Run an interactive chat session with the Oakie assistant."""
    args = process_command_line_args()
    
    print("Initializing Oakie Technologies AI Assistant...")
    
    try:
        assistant = OakieAssistant()
        print(f"Welcome to {assistant.guidelines.company_name} AI Assistant!")
        
        # Load documents if provided
        if args.docs:
            print("Loading provided documents...")
            for doc_path in args.docs:
                if assistant.load_document_from_file(doc_path):
                    print(f"Loaded: {os.path.basename(doc_path)}")
                else:
                    print(f"Failed to load: {doc_path}")
        
        # Non-interactive mode: process a single query
        if args.non_interactive:
            if args.query:
                print(f"\nQuery: {args.query}")
                response = assistant.ask(args.query)
                print(f"\nAssistant: {response}")
                return
            else:
                print("Error: Non-interactive mode requires a query. Use --query or -q.")
                return
        
        # Interactive mode
        print("Type 'exit' to quit, 'reset' to start a new conversation, or use commands:")
        print("  /doc [filepath] - Load a document")
        print("  /docs - List loaded documents")
        print("  /cleardocs - Clear loaded documents")
        print("  /mentions [message indices] - Reference previous messages (e.g., /mentions 1,3,5)")
        
        chat_history = []  # Store all interactions for potential reference
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'exit':
                print("Thank you for using the Oakie Technologies AI Assistant. Goodbye!")
                break
                
            if user_input.lower() == 'reset':
                assistant.reset_conversation()
                chat_history = []
                print("Conversation has been reset.")
                continue
            
            # Command processing
            if user_input.startswith('/doc '):
                doc_path = user_input[5:].strip()
                if assistant.load_document_from_file(doc_path):
                    print(f"Document loaded: {os.path.basename(doc_path)}")
                else:
                    print(f"Failed to load document: {doc_path}")
                continue
                
            if user_input == '/docs':
                if assistant.documents:
                    print("Loaded documents:")
                    for i, doc in enumerate(assistant.documents.keys()):
                        print(f"  {i+1}. {doc}")
                else:
                    print("No documents loaded.")
                continue
                
            if user_input == '/cleardocs':
                assistant.documents = {}
                print("All documents cleared.")
                continue
                
            if user_input.startswith('/mentions '):
                try:
                    indices = [int(idx) for idx in user_input[10:].strip().split(',')]
                    mentions = []
                    for idx in indices:
                        if 0 < idx <= len(chat_history):
                            mentions.append(chat_history[idx-1])
                    
                    if not mentions:
                        print("No valid message indices provided.")
                        continue
                        
                    print(f"You'll be asked for your query next, with {len(mentions)} referenced messages.")
                    user_query = input("\nYour query with mentions: ")
                    response = assistant.ask(user_query, mentions=mentions)
                    
                except ValueError:
                    print("Invalid message indices. Use format: /mentions 1,3,5")
                    continue
            else:
                # Regular query
                response = assistant.ask(user_input)
                
            print(f"\nAssistant: {response}")
            
            # Store the interaction in chat history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
            
    except KeyboardInterrupt:
        print("\nSession terminated by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    main()
