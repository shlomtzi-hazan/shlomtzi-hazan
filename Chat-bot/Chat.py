from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import load_tools
from langchain.tools import Tool

# Load environment variables from .env file
load_dotenv()

# Retrieve variables from the environment
api_key = os.getenv("OPENAI_API_KEY")
searx_host = os.getenv("SEARX_HOST")
searx_engines = os.getenv("SEARX_ENGINES").split(",")
model = os.getenv("OPENAI_MODEL")

tools = load_tools(
    ["searx-search"],
    searx_host=searx_host,
    engines=searx_engines
)

def conversetional_input(user_input):
    print(f"Searching for: {user_input}")
    return llm.invoke(user_input)

tools.append(
    Tool(
        name="Conversational Input",
        func=conversetional_input,
        description="This tool is used to provide a conversational response based on the LLM. It is invoked when the user query doesn't require an online search (e.g., general knowledge, basic information, or pre-existing knowledge within the model)."
    )
)

llm = ChatOpenAI(api_key=api_key, model=model)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

def chat():
    print("Chat started! Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Ending chat.")
            break

        response = agent.invoke(user_input)
        try:
            if isinstance(response, dict):
                if 'output' in response:
                    clean_response = response['output'].replace("\n", " ").strip()
                else:
                    clean_response = str(response)
            else:
                clean_response = response.replace("\n", " ").strip()
            print(f"OpenAI: {clean_response.strip()}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat()
