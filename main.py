from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from googletrans import Translator, LANGUAGES
import model_define
import dataloading
import torch
import traceback
import uvicorn
import os

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
model = model_define.Model()
model.load_state_dict(torch.load("final_model.pth"))
model.eval()
print("Model Loaded successfully...")

translator = Translator()
response = {}

class SentimentRequest(BaseModel):
    text: str

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_sentiment(request: SentimentRequest):
    try:
        text = request.text
        detected_lang = translator.detect(text).lang
        translated_text = text if detected_lang == "en" else translator.translate(text, src=detected_lang, dest="en").text
        embeddings = dataloading.embed_text([translated_text])

        if embeddings is None:
            raise ValueError("Embedding function returned None.")

        embeddings_tensor = torch.Tensor(embeddings)
        logits = model(embeddings_tensor)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        scores = torch.softmax(logits, dim=1).detach().cpu().numpy()

        best_index = preds[0]
        sentiment = "positive" if best_index == 1 else "negative"
        confidence = float(scores[0][best_index])  

        response.update({
            "original_text": text,
            "translated_text": translated_text if detected_lang != "en" else None,
            "sentiment": sentiment,
            "confidence": confidence,
            "detected_language": LANGUAGES.get(detected_lang, "unknown")
        })

        return JSONResponse(content=response)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/result")
async def get_result():
    return JSONResponse(content=response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)