from fastapi import FastAPI
import joblib

app = FastAPI()

vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")


@app.get("/")
def home():
    return {"message": "Comments API is running"}


@app.post("/predict")
def predict(data: dict):
    text = data.get("text", "")

    if not text.strip():
        return {"result": "الرجاء إدخال نص صالح للتحليل."}

    transformed = vectorizer.transform([text])

    prediction = str(model.predict(transformed)[0])

    result = (
        "سلبي (Negative)"
        if prediction == "0"
        else "إيجابي/آخر (Positive)"
    )

    return {"result": result}