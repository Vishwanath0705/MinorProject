from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from your frontend (http://127.0.0.1:5501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5501"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.post("/predict/")
async def predict(data: dict):
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}
    return {"label": "Positive", "score": 0.95}  # Dummy response
