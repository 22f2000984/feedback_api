from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os

app = FastAPI(title="Customer Feedback Analysis API")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Request & Response Models ----------

class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1, description="Customer comment")

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., pattern="^(positive|negative|neutral)$")
    rating: int = Field(..., ge=1, le=5)

# ---------- Endpoint ----------

@app.post(
    "/comment",
    response_model=SentimentResponse,
    response_model_exclude_none=True,
)
def analyze_comment(request: CommentRequest):
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis engine. "
                        "Classify the sentiment and return a rating.\n\n"
                        "Rules:\n"
                        "- positive → rating 4 or 5\n"
                        "- neutral → rating 3\n"
                        "- negative → rating 1 or 2\n"
                        "- Return ONLY valid JSON matching the schema."
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
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

        return response.output_parsed

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}"
        )