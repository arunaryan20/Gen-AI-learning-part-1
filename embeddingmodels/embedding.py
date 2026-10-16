from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    dimensions=64
)

texts=[
    "Hello, this is aryan",
    "Hello your name is Gen AI",
    "And you all are very beautiful people"
]

vector =embeddings.embed_documents(texts)

# vector =embeddings.embed_query("You are going to learn Gen AI")

print(vector)
