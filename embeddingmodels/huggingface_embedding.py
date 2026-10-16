from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

text=["Cricket is a popular bat and ball sport"]

vector =embeddings.embed_documents(text)

print(vector)