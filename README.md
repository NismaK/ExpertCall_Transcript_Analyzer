# Expert Call Transcript Analyzer

A single-file Streamlit app that reads 3 expert-call transcripts, answers a set
of interview-guide questions per expert with cited quotes and timestamps,
surfaces cross-expert themes/disagreements, and answers free-form questions
across all three.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Get a free Groq API key at console.groq.com/keys, paste it into the sidebar
(or set it as a `GROQ_API_KEY` secret so it's pre-filled), upload the 3
transcripts, and click Analyze. The interview-guide questions box is
pre-filled with the 6 questions from `Interview_Guide.txt` but stays editable.

Two transcript formats are supported out of the box:
- Inline: `[00:01:23] Speaker: text`
- Block (matches the actual case transcripts): a timestamp alone on its own
  line, e.g. `00:18`, followed by `Speaker: text` on the next line. Any
  lines before the first timestamp (name / role / market) are captured as
  header info and shown at the top of that expert's tab.

## Architecture

```
3 .txt transcripts
      │
      ▼
parse_transcript()  →  structured segments: {timestamp, speaker, text}
      │
      ▼
Groq API (openai/gpt-oss-120b), one call per task:
  1. extract_answers()   → per-expert answers to interview-guide questions
  2. analyze_themes()    → common themes + disagreements across all 3
  3. ask_question()      → free-form Q&A across all 3
      │
      ▼
Post-processing: verify every quote is an exact substring of the source
transcript before showing it as "verified"
      │
      ▼
Streamlit UI: tabs per expert + cross-expert tab + chat-style Q&A tab
```

No database, no vector store, no backend server — everything runs in one
Streamlit process for a 3-transcript demo. That's a deliberate simplicity
choice explained below.

## Model choice

**Groq, running `openai/gpt-oss-120b`.**

Despite the `openai/` prefix, this is an **open-weight model hosted on Groq's
free tier** (OpenAI's own open-weight release, not a call to OpenAI's paid
API). Groq's own catalog previously recommended `llama-3.3-70b-versatile` for
this kind of task, but Groq deprecated and decommissioned that model on
August 16, 2026 — `openai/gpt-oss-120b` (or `qwen/qwen3.6-27b`) is their
current recommended, free-tier replacement.

- Free tier, generous rate limits, good enough quality for structured
  extraction over a few thousand words of transcript.
- Large context window, so a full transcript (or all 3 combined) fits in one
  call — no chunking or retrieval needed at this scale.
- Groq's inference is very fast (LPU hardware), which matters for a live demo
  where you re-run analysis in front of an interviewer.
- I asked the model to return `response_format={"type": "json_object"}` so
  answers come back structured, not as free text I have to regex-parse.

Alternative I'd mention in the interview: **Gemini 2.5 Flash** — also free
tier, 1M-token context window, so it's the natural upgrade path if
transcripts get much longer than Groq's context comfortably handles.
Swapping providers only touches the `call_llm()` function. Neither this nor
the primary model requires a paid account.

## How citations/timestamps work

Transcripts are parsed line-by-line into `{timestamp, speaker, text}`
segments before anything is sent to the model. The full transcript is then
re-serialized with timestamps still attached (`[00:01:23] Speaker: text`) and
given to the model as context. The model is instructed to return, for every
answer, the exact timestamp line the answer came from plus a verbatim quote —
so every claim is traceable back to a specific moment in a specific
transcript.

## How hallucinations are reduced

1. **Strict system prompt** — explicitly told to only use what's in the
   transcript, and to say "Not found in transcript" rather than guess.
2. **Temperature 0** — deterministic, less creative completion.
3. **Structured JSON output** — forces the model into a fixed schema instead
   of free-flowing prose, which reduces drift.
4. **Post-hoc quote verification** — after the model responds, the app checks
   whether each returned quote is an actual substring of the source
   transcript (whitespace-normalized). If it isn't, it's flagged
   "⚠️ unverified" in the UI instead of being silently trusted. This is a
   cheap, deterministic guardrail that doesn't need another model call.

## Scaling from 3 transcripts to 30+

The current design stuffs full transcripts into the prompt, which works up to
maybe ~10 transcripts before hitting context/cost/latency limits. At 30+:

- **Chunk + embed** each transcript into ~500-token segments (keeping
  timestamp metadata attached to each chunk).
- **Vector store** (e.g. Chroma, pgvector, or Pinecone) for semantic
  retrieval instead of full-context stuffing.
- **Retrieval-augmented answering**: for each interview-guide question, embed
  the question, pull top-k relevant chunks per expert, and only send those to
  the model — keeps cost/latency flat as transcript count grows.
- **Map-reduce for cross-expert themes**: summarize each transcript
  independently first (map), then run theme/disagreement analysis over the
  summaries (reduce), rather than concatenating 30 raw transcripts into one
  prompt.
- Same quote-verification step still applies — it just checks against the
  retrieved chunk instead of the full transcript.

## Notes

- No database is used; everything lives in `st.session_state` for the
  session. For a real product you'd persist parsed transcripts and answers
  so re-analysis isn't needed every time.
- The interview-guide questions are editable in the UI rather than
  hardcoded, since the actual guide wasn't provided in the case.
