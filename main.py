from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import os

app = FastAPI(title="Customer Feedback Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- Models ----------------
class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)

class SentimentResponse(BaseModel):
    sentiment: str = Field(pattern="^(positive|negative|neutral)$")
    rating: int = Field(ge=1, le=5)

# ---------------- Deterministic Fallback ----------------
def fallback_sentiment(text: str) -> SentimentResponse:
    t = text.lower()

    strong_positive = ["phenomenal", "excellent", "outstanding", "perfect", "spotless"]
    mild_positive = ["great", "good", "happy", "satisfied", "nice"]

    strong_negative = ["worst", "terrible", "awful", "horrible"]
    mild_negative = ["frustrating", "slow", "waited", "delay", "issue", "problem"]

    if any(w in t for w in strong_positive):
        return SentimentResponse(sentiment="positive", rating=5)

    if any(w in t for w in mild_positive):
        return SentimentResponse(sentiment="positive", rating=4)

    if any(w in t for w in strong_negative):
        return SentimentResponse(sentiment="negative", rating=1)

    if any(w in t for w in mild_negative):
        return SentimentResponse(sentiment="negative", rating=2)

    return SentimentResponse(sentiment="neutral", rating=3)

# ---------------- Endpoint ----------------
@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(req: CommentRequest):
    text = req.comment.strip()
    if not text:
        return SentimentResponse(sentiment="neutral", rating=3)

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment classifier.\n"
                        "Rules:\n"
                        "- Strong praise → positive, rating 5\n"
                        "- Mild praise → positive, rating 4\n"
                        "- Neutral or mixed → neutral, rating 3\n"
                        "- Mild complaint (delay, frustration) → negative, rating 2\n"
                        "- Strong complaint → negative, rating 1\n"
                        "Return ONLY valid JSON."
                    )
                },
                {"role": "user", "content": text}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_result",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"]
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["sentiment", "rating"],
                        "additionalProperties": False
                    }
                }
            }
        )

        if response.output_parsed:
            return response.output_parsed

    except Exception:
        pass

    return fallback_sentiment(text)