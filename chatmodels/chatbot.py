from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

model = ChatOllama(model="qwen2.5:7b")

print("Choose your AI mode")
print("Press 1 for Angry Mode")
print("Press 2 for Funny Mode")
print("Press 3 for Sad Mode")

choice = int(input("Tell your response :"))

if choice == 1:
    mode = "You are an angry AI agent. You respond aggressively and impatiently"
elif choice == 2:
    mode = "You are a very funny AI agent. You respond with homor and jokes"
elif choice == 3:
    mode = "You are a sad AI agent. You respond with empathy and understanding"


messages = [SystemMessage(content=mode)]

prompt = "You : "
print("--------------Type 'exit' to quit the chat.------------------------")
while prompt != "exit":
    prompt = input("You : ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "exit":
        break

    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Assistant :", response.content)
