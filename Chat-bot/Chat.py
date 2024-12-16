from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import load_tools
from langchain.tools import Tool

tools = load_tools(["searx-search"],
                    searx_host="https://eladsearngx-w4o7wofqcq-nn.a.run.app/",
                    engines=["google"])

def conversetional_input(user_input):
    """Fallback tool: This tool provides conversational responses directly from the LLM for user queries that do not require real-time information or online search, such as general knowledge or static information already contained within the model."""
    # Criterion for invoking this tool
    print(f"Searching for: {user_input}")
    return llm.invoke(user_input)

tools.append(
    Tool(
        name="Conversational Input",
        func=conversetional_input,
        description="This tool is used to provide a conversational response based on the LLM. It is invoked when the user query doesn't require an online search (e.g., general knowledge, basic information, or pre-existing knowledge within the model)."
    )
)

with open("/Users/shlomtzi/PycharmProjects/api_key.txt", "r") as file:
    api_key = file.read().strip()

llm = ChatOpenAI(api_key=api_key, model="gpt-4-turbo")
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

def chat():
    # Start the conversation
    print("Chat started! Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Ending chat.")
            break
        
        # Get the response from the agent
        response = agent.invoke(user_input)
       
        try:
            # Check if response is a dictionary and print the keys
            if isinstance(response, dict):
                # Access the 'output' key to get the model's response
                if 'output' in response:
                    clean_response = response['output'].replace("\n", " ").strip()
                else:
                    clean_response = str(response)  # Fallback if 'output' key is missing
            else:
                # If the response is a string, clean it directly
                clean_response = response.replace("\n", " ").strip()
            
            print(f"OpenAI: {clean_response.strip()}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    chat()
