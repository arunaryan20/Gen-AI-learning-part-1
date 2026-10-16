from dotenv import load_dotenv

load_dotenv()

import os

# model class implementation
# from langchain_openai import ChatOpenAI

# model = ChatOpenAI(model="gpt-5.5")

# response = model.invoke("what is cricket?")
# print(response.content)


# init chat model implementation
from langchain.chat_models import init_chat_model

# open ai model
model = init_chat_model("gpt-5.5",temperature=0.7, max_tokens=20)

# gemini ai model
# model = init_chat_model("google_genai:gemini-3.7-flash")

response = model.invoke("what is cricket?")
print(response.content)
