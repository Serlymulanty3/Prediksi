#set api key sekali di env melalui terminal: (setx GEMINI_API_KEY "AIzaSyAUARjIjm2JSoYI0UqSva7MlIFWmrIKupQ")
#api key gemini saya "AIzaSyAUARjIjm2JSoYI0UqSva7MlIFWmrIKupQ"

import os, google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = "gemini-2.5-flash"  # atau "gemini-flash-latest"
m = genai.GenerativeModel(MODEL_NAME)

r = m.generate_content("Halo Gemini! Jawab singkat: 3 + 4 = ?")
print("MODEL:", MODEL_NAME)
print("RESP :", r.text[:200])
