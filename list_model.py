import os, google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# tampilkan model yang mendukung generateContent
for m in genai.list_models():
    if "generateContent" in getattr(m, "supported_generation_methods", []):
        print(m.name)
