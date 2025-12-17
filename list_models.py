import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv("server/.env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("GOOGLE_API_KEY not found")
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Listing models...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"Error: {e}")
