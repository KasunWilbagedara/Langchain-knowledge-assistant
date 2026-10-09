
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not loaded from .env")

client = genai.Client(api_key=api_key)

print("Available Gemini models:\n")

for model in client.models.list():
    actions = getattr(model, "supported_actions", []) or []

    if "generateContent" in actions:
        print(model.name)
