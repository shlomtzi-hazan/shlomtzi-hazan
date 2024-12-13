from langchain_openai import ChatOpenAI

with open("/Users/shlomtzi/PycharmProjects/api_key.txt", "r") as file:
    api_key = file.read().strip()

llm = ChatOpenAI(api_key=api_key)

# Start the conversation
print("Chat started! Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Ending chat.")
        break
    response = llm.invoke(user_input)

    if hasattr(response, 'content'):
        clean_response = response.content  # Extract only the content part
        # Clean up the content if needed (remove newlines or extra spaces)
        clean_response = clean_response.replace("\n", " ").strip()
    else:
        clean_response = response

    print(f"OpenAI: {clean_response.strip()}")
