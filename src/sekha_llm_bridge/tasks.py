from typing import List

import litellm

from .config import settings
from .workers.celery_app import celery_app


@celery_app.task(name="tasks.embed_text")
def embed_text_task(text: str, model: str | None = None) -> list[float]:
    model_name = model or settings.embedding_model
    # Use LiteLLM embeddings where supported; fall back to simple OpenAI format
    response = litellm.embedding(
        model=model_name,
        input=text,
    )
    # LiteLLM returns OpenAI-style structure
    return response["data"][0]["embedding"]


@celery_app.task(name="tasks.summarize_messages")
def summarize_messages_task(
    messages: List[str], level: str, model: str | None = None
) -> str:
    model_name = model or settings.summarization_model
    joined = "\n".join(messages)

    system_prompt = (
        "You are a summarization assistant.\n"
        f"Generate a concise {level} summary.\n"
        "Focus on key decisions, action items, and facts.\n"
        "Limit to 3-6 bullet points or 2-3 sentences."
    )

    completion = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": joined},
        ],
        max_tokens=256,
        temperature=0.2,
    )

    return completion.choices[0].message["content"]


@celery_app.task(name="tasks.extract_entities")
def extract_entities_task(text: str, model: str | None = None) -> list[dict]:
    model_name = model or settings.extraction_model

    system_prompt = (
        "Extract named entities from the text as JSON with fields: "
        "[{type, value, confidence}]. Types may include PERSON, ORG, LOCATION, DATE, TOPIC."
    )

    completion = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        max_tokens=256,
        temperature=0.0,
    )

    import json

    content = completion.choices[0].message["content"]
    try:
        entities = json.loads(content)
        if isinstance(entities, list):
            return entities
        return []
    except Exception:
        return []


@celery_app.task(name="tasks.score_importance")
def score_importance_task(text: str, model: str | None = None) -> float:
    model_name = model or settings.summarization_model

    system_prompt = (
        "You are an importance scoring assistant. "
        "Given a message, return a single number between 1 and 10 "
        "indicating how important it is to remember for future context. "
        "10 = critical decisions or long-term goals, 1 = trivial small talk. "
        "Respond with only the number."
    )

    completion = litellm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        max_tokens=8,
        temperature=0.0,
    )

    content = completion.choices[0].message["content"].strip()
    try:
        score = float(content)
    except ValueError:
        score = 5.0

    score = max(1.0, min(10.0, score))
    return score
