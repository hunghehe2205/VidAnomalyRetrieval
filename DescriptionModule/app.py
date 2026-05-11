import re
import json
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Video Summary Service")

LLM_BASE_URL = "http://localhost:8000/v1"
LLM_API_KEY = "EMPTY"
LLM_MODEL = "google/gemma-3-4b-it"

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)

# These are generation preferences only. They are intentionally NO`T used
# as hard validation rules, because style violations should not break batch jobs.
DISCOURAGED_PHRASES = [
    "anomaly", "abnormal", "suggesting", "appears", "potentially",
    "involves", "seems", "possibly", "indicating", "the event",
    "the incident", "the scene shows",
]

SYSTEM_PROMPT = """You are a video summarizing assistant.
Your task is to merge the global video caption and clip captions into a full video-level natural-language description and a concise summary.

Return valid JSON only. Do not include markdown, bullet points, lists, or any preamble.

1. full_summary:
- Describe the visible activity across the whole video.
- Use 1 to 4 clear sentences.
- Merge the global caption and all clip captions.
- Preserve temporal flow when possible.
- Remove duplicate details.
- Avoid inventing people, objects, locations, clothing, or actions.
- If captions disagree about visual details, use general wording such as "light clothing", "dark clothing", or "a person".
- Do not speculate or classify unverified intent.

2. summary:
- Prefer 1 sentence and 25 to 30 words when it reads naturally.
- Use a maximum of 2 sentences when needed for clarity.
- If 25 to 30 words is not natural, still produce a clear summary longer than 10 words.
- Use past tense.
- Use declarative style.
- Use third person.
- Often start with time or location when available, such as "At night," "In the parking lot," "On the street," or "Near the vehicle,".
- Include subjects with visual attributes when available, such as "a man in a black hat" or "two men wearing helmets".
- Mention the main visible action.
- Avoid inventing attributes not present in the captions.

3. DEFAULT MODE:
Use default mode when the global caption and clip captions agree on what happened.
- State the action directly using concrete verbs when supported by the captions, such as beat, kicked, shot, lit, smashed, snatched, robbed, hit, set fire, broke in, or stole.
- Describe what happened factually.
- Do not overstate actions that are only present in the global caption but not supported by clips.

4. CONFLICT MODE:
Use conflict mode when clips strongly contradict the global caption, clips describe normal behavior while the global caption describes a crime, clips describe completely different people/actions, or clips do not support the claimed harmful/criminal action.
- Do not state the contested harmful, criminal, or suspicious action.
- Describe only the visible scene and visible movement.
- Use neutral verbs such as walked, stood, moved, ran, approached, gathered, sat, entered, or left.
- Skip the contested action entirely.
- Do not invent or classify.

5. STYLE PREFERENCES:
Avoid these words or phrases when possible in both full_summary and summary:
anomaly, abnormal, suggesting, appears, potentially, involves, seems, possibly, indicating, the event, the incident, the scene shows.
Avoid speculation, hedging, bullet points, lists, and preambles such as "Here is" or "Summary:".

Examples, default mode:
- "A man in a hat beat a white dog in the yard with a stick outside the railing."
- "At night, a man set fire next to a car on the side of the road."
- "In the parking lot at night, a man shot at several others."
- "A man in a white shirt kicked the door open with his foot and entered the house."

Example, conflict mode:
Global: "Two men attacked a third man in a parking lot."
Clips: "A man stood near a car." / "Several people gathered in a parking lot." / "A man walked toward the camera."
Output: "Several men gathered near a car in a parking lot at night."

JSON schema:
{
  "full_summary": "string",
  "summary": "string",
  "anomaly_type": "string"
}
"""

CORRECTION_PROMPT = """The previous output was invalid JSON, missed required fields, or produced a very short summary.
Error details: {error_details}

Please correct the output and return ONLY valid JSON matching this schema:
{
  "full_summary": "string",
  "summary": "string",
  "anomaly_type": "string"
}

The summary should be clear, should contain more than 10 words when possible, and should use no more than 2 sentences when possible.
Prefer the requested style, but do not force an exact word count.
"""

class Clip(BaseModel):
    frame_range: List[int]
    prompt: str
    caption: str

class VideoRequest(BaseModel):
    video: str
    fps: float
    num_frames: int
    video_prompt: str
    video_caption: str
    clips: List[Clip]

class SummaryResponse(BaseModel):
    video: str
    full_summary: str
    summary: str
    anomaly_type: str

def required_field_errors(data: dict) -> List[str]:
    if not isinstance(data, dict):
        return ["Output is not a JSON object"]
    required_fields = ["full_summary", "summary", "anomaly_type"]
    missing = [field for field in required_fields if not data.get(field)]
    if missing:
        return [f"Missing required fields: {', '.join(missing)}"]
    return []

def soft_style_errors(data: dict) -> List[str]:
    """Return style issues that may trigger one retry but must not fail the API."""
    errors = []
    summary = data.get("summary", "") if isinstance(data, dict) else ""

    words = re.findall(r"\b\w+\b", summary)
    word_count = len(words)
    if word_count <= 10:
        errors.append(
            f"Summary word count is {word_count}. It should be longer than 10 words when possible."
        )

    return errors

def parse_json_response(content: str):
    if not content:
        return None, "Empty response from LLM"

    content = content.strip()
    match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
    if match:
        content = match.group(1).strip()

    try:
        data = json.loads(content)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {str(e)}"

def call_llm(messages: list):
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)

def build_user_prompt(request: VideoRequest) -> str:
    prompt = f"Video path: {request.video}\n"
    prompt += f"Video global caption: {request.video_caption}\n\nClip captions:\n"
    for idx, clip in enumerate(request.clips, 1):
        prompt += f"- Clip {idx} (Frames {clip.frame_range[0]}-{clip.frame_range[1]}): {clip.caption}\n"

    prompt += "\nBased on the above global caption and clip captions, provide full_summary, summary, and anomaly_type according to the instructions."
    return prompt

def retry_llm(messages: list, previous_response: str, error_details: str):
    correction_msg = CORRECTION_PROMPT.format(error_details=error_details)
    retry_messages = messages + [
        {"role": "assistant", "content": previous_response or ""},
        {"role": "user", "content": correction_msg},
    ]
    return call_llm(retry_messages)

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_video(request: VideoRequest):
    prompt = build_user_prompt(request)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response_text, error_msg = call_llm(messages)
    if error_msg:
        raise HTTPException(status_code=500, detail=f"LLM API failure: {error_msg}")

    parsed, parse_err = parse_json_response(response_text)

    # Hard issue: malformed JSON. Retry once, then fail only if JSON is still malformed.
    if parse_err:
        response_text2, error_msg2 = retry_llm(messages, response_text, parse_err)
        if error_msg2:
            raise HTTPException(status_code=500, detail=f"LLM API failure on retry: {error_msg2}")
        parsed2, parse_err2 = parse_json_response(response_text2)
        if parse_err2:
            raise HTTPException(status_code=500, detail=f"Retry failed to return valid JSON: {parse_err2}")
        parsed = parsed2
        response_text = response_text2

    # Hard issue: required fields. Retry once, then fail only if fields are still missing.
    missing_errors = required_field_errors(parsed)
    if missing_errors:
        response_text2, error_msg2 = retry_llm(messages, response_text, "; ".join(missing_errors))
        if error_msg2:
            raise HTTPException(status_code=500, detail=f"LLM API failure on retry: {error_msg2}")
        parsed2, parse_err2 = parse_json_response(response_text2)
        if parse_err2:
            raise HTTPException(status_code=500, detail=f"Retry failed to return valid JSON: {parse_err2}")
        missing_errors2 = required_field_errors(parsed2)
        if missing_errors2:
            raise HTTPException(status_code=500, detail=f"Retry still missing required fields: {'; '.join(missing_errors2)}")
        parsed = parsed2
        response_text = response_text2

    # Soft issue: summary too short. Retry once, but return valid output even if still short.
    style_errors = soft_style_errors(parsed)
    if style_errors:
        response_text2, error_msg2 = retry_llm(messages, json.dumps(parsed, ensure_ascii=False), "; ".join(style_errors))
        if not error_msg2:
            parsed2, parse_err2 = parse_json_response(response_text2)
            if not parse_err2 and not required_field_errors(parsed2):
                parsed = parsed2

    return SummaryResponse(
        video=request.video,
        full_summary=parsed["full_summary"],
        summary=parsed["summary"],
        anomaly_type=parsed["anomaly_type"],
    )
