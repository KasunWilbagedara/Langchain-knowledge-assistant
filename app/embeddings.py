from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY was not loaded from .env")

embeddings = GoogleGenerativeAIEmbeddings(
  model="gemini-embedding-001",
  google_api_key=api_key
)

vector = embeddings.embed_query("what is rag?")
print("Vector length:", len(vector))
print("First 10 values:", vector[:10])