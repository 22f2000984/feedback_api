from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import os

# ---------------- App ----------------
app = FastAPI(title="Customer Feedback Analysis API")

# CORS (prevents OPTIONS 405 in evaluators)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client (ENV VAR ONLY)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- Models ----------------
class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)

class SentimentResponse(BaseModel):
    sentiment: str = Field(pattern="^(positive|negative|neutral)$")
    rating: int = Field(ge=1, le=5)

# ---------------- Endpoint ----------------
@app.post("/comment", response_model=SentimentResponse)
def analyze_comment(req: CommentRequest):
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=req.comment,
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

        return response.output_parsed

    except Exception as e:
        # Graceful failure (required by problem)
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")