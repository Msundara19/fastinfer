"""
Groq LLM analysis for prediction results.
Generates plain-English explanations of model predictions for non-technical users.
"""

import os

try:
    from groq import AsyncGroq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


async def analyze_prediction(
    class_name: str,
    confidence: float,
    latency_ms: float,
    model: str
) -> str:
    """
    Generate a plain-English analysis of a prediction result using Groq.
    Returns a fallback string if Groq is unavailable or the API key is not set.
    """
    api_key = os.environ.get("GROQ_API_KEY")

    if not _GROQ_AVAILABLE or not api_key:
        return None

    client = AsyncGroq(api_key=api_key)

    conf_pct = round(confidence * 100, 1)
    conf_level = "high" if conf_pct >= 70 else "moderate" if conf_pct >= 40 else "low"
    readable_class = class_name.replace("_", " ")

    prompt = (
        f'An image classifier identified the subject as "{readable_class}" '
        f"with {conf_pct}% confidence ({conf_level} confidence) "
        f"using the {model} backend in {latency_ms}ms.\n\n"
        f"Write 2-3 friendly sentences for a general audience: "
        f"(1) what was detected and whether that seems reasonable, "
        f"(2) what {conf_pct}% confidence means in practice, "
        f"(3) a brief note on the {latency_ms}ms inference speed. "
        f"Be conversational — no ML jargon."
    )

    response = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=180,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()
