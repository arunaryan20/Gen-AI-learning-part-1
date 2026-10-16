# ----------------------------------Output Parser----------------------------------

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser


class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: Optional[str]
    rating: Optional[float]
    summary: Optional[str]


parser = PydanticOutputParser(pydantic_object=Movie)
model = ChatOllama(model="qwen2.5:7b", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an information extraction and summarization assistant.

Analyze the paragraph provided by the user and extract the most useful information from it.

{format_instructions}

Rules:
- Extract information only from the provided paragraph.
- Do not invent information that is not present.
- Correct obvious spelling mistakes when the intended name or word is clear.
- If some information is not available, set the field to null (or use "Not mentioned" for text fields).
- Keep the summary to 2-3 sentences.
- Make the output valid JSON that strictly matches the schema.
- Do NOT include any explanation, comments, or text outside the JSON.
            """,
        ),
        ("human", "{text}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


print("--------------------------------------------------")
print("Enter your paragraph below.")
print("Type 'exit' to quit.")
print("--------------------------------------------------")

while True:
    paragraph = input("\nYou: ")

    if paragraph.lower() == "exit":
        print("Goodbye!")
        break

    messages = prompt.invoke({"text": paragraph})

    response = model.invoke(messages)

    print("\nAssistant (raw):")
    print(response.content)

    # Parse the model output into the Movie pydantic object
    try:
        movie: Movie = parser.parse(response.content)
        print("\nParsed Movie object:")
        print(movie)
    except Exception as e:
        print("\nFailed to parse output:", e)



# ----------------------------------Prompt Template----------------------------------

# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from pydantic import BaseModel
# from typing import List, Optional
# from langchain_core.output_parsers import PydanticOutputParser


# class Movie(BaseModel):
#     title: str
#     release_year: Optional[int]
#     genre: List[str]
#     director: Optional[str]
#     cast: Optional[str]
#     rating: Optional[float]
#     summary: Optional[str]


# parser = PydanticOutputParser(pydantic_object=Movie)
# model = ChatOllama(model="qwen2.5:7b", temperature=0)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
# You are an information extraction and summarization assistant.

# Analyze the paragraph provided by the user and extract the most useful information from it.

# Provide the answer in the following format:

# Title:
# Director:
# Release Year:
# Genre:
# Cast:
# Plot:
# Setting:
# Themes:
# Soundtrack:
# IMDb Rating:
# Reception:
# Key Points:
# Quick Summary:

# Rules:
# - Extract information only from the provided paragraph.
# - Do not invent information that is not present.
# - Correct obvious spelling mistakes when the intended name or word is clear.
# - If some information is not available, write "Not mentioned".
# - Keep the Plot concise.
# - Keep the Quick Summary to 2-3 sentences.
# - Key Points should contain the most important facts from the paragraph.
# - Make the output clear and easy to read.
#         """,
#         ),
#         ("human", "{text}"),
#     ]
# )


# print("--------------------------------------------------")
# print("Enter your paragraph below.")
# print("Type 'exit' to quit.")
# print("--------------------------------------------------")

# while True:
#     paragraph = input("\nYou: ")

#     if paragraph.lower() == "exit":
#         print("Goodbye!")
#         break

#     messages = prompt.invoke({"text": paragraph})

#     response = model.invoke(messages)

#     print("\nAssistant:")
#     print(response.content)


# -----------------------Normal-----------------------

# from langchain_ollama import ChatOllama
# from langchain_core.prompts import PromptTemplate

# model = ChatOllama(model="qwen2.5:7b")

# prompt=PromptTemplate.from_template("")

# response = model.invoke("Hello")

# print(response.content)



