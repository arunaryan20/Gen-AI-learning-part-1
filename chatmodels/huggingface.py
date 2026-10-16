# import os
# from dotenv import load_dotenv

# load_dotenv()

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# llm = HuggingFaceEndpoint(repo_id="Aleph-Alpha/Kolibri-1")

# model = ChatHuggingFace(llm=llm)

# response = model.invoke("What is cricket?")
# print(response.content)


#  Using local model
from langchain_ollama import ChatOllama

model =ChatOllama(model="qwen2.5:7b")

response =model.invoke("write a java program to add two numbers")

print(response.content)
