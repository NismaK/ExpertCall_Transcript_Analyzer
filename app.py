"""
Expert Call Transcript Analyzer
Upload 3 expert-call transcripts, get per-expert answers to interview-guide
questions, extracted quotes with timestamps, cross-expert themes/disagreements,
and free-form Q&A across all transcripts.
"""

import json
import re
import streamlit as st
from groq import Groq

MODEL = "openai/gpt-oss-120b"  # Groq's free-tier replacement for the deprecated llama-3.3-70b-versatile

# ---------------------------------------------------------------------------
# 1. Parsing: turn raw transcript text into timestamped segments
# ---------------------------------------------------------------------------
# Supports two formats:
#   A) inline:   [00:01:23] Speaker: text
#   B) block:    00:18
#                Dr. Martin: text
# Any lines before the first timestamp (name/role/market header) are captured
# separately and shown as context, not treated as dialogue.
BRACKETED = re.compile(r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*([^:]+):\s*(.+)$")
TIMESTAMP_ONLY = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)$")
SPEAKER_LINE = re.compile(r"^([^:]+):\s*(.+)$")


def parse_transcript(raw_text: str):
    """Return (header_text, segments) where segments is a list of
    {timestamp, speaker, text} dicts."""
    header_lines = []
    segments = []
    pending_ts = None
    seen_timestamp = False

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = BRACKETED.match(line)
        if m:
            ts, speaker, text = m.groups()
            segments.append({"timestamp": ts, "speaker": speaker.strip(), "text": text.strip()})
            pending_ts = None
            seen_timestamp = True
            continue

        m = TIMESTAMP_ONLY.match(line)
        if m:
            pending_ts = m.group(1)
            seen_timestamp = True
            continue

        m = SPEAKER_LINE.match(line)
        if m and pending_ts:
            speaker, text = m.groups()
            segments.append({"timestamp": pending_ts, "speaker": speaker.strip(), "text": text.strip()})
            pending_ts = None
            continue

        if not seen_timestamp:
            # part of the header block (name / role / market) before dialogue starts
            header_lines.append(line)
        elif segments:
            # continuation of the previous spoken line
            segments[-1]["text"] += " " + line

    return "\n".join(header_lines), segments


def transcript_as_numbered_text(segments):
    """Render segments as a numbered, timestamped block for the LLM prompt."""
    return "\n".join(f"[{s['timestamp']}] {s['speaker']}: {s['text']}" for s in segments)


# ---------------------------------------------------------------------------
# 2. LLM calls
# ---------------------------------------------------------------------------
def call_llm(client, system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


ANSWER_SYSTEM_PROMPT = """You are a careful research assistant analyzing a single \
expert-call transcript. You must ONLY use information present in the transcript. \
Never invent facts, numbers, or opinions the expert did not state. \
For every question, return an exact quote copied verbatim from the transcript \
(do not paraphrase the quote) and the timestamp that precedes it. \
If the transcript does not address a question, set "answer" to \
"Not found in transcript" and leave "quote" and "timestamp" empty. \
Respond ONLY with JSON of this shape:
{"answers": [{"question": "...", "answer": "...", "quote": "...", "timestamp": "..."}]}"""


def extract_answers(client, segments, questions):
    transcript_text = transcript_as_numbered_text(segments)
    user_prompt = (
        f"TRANSCRIPT:\n{transcript_text}\n\n"
        f"QUESTIONS:\n" + "\n".join(f"- {q}" for q in questions)
    )
    raw = call_llm(client, ANSWER_SYSTEM_PROMPT, user_prompt)
    data = json.loads(raw)
    answers = data.get("answers", [])
    # Hallucination check: verify each quote actually appears in the transcript
    for a in answers:
        quote = (a.get("quote") or "").strip()
        a["verified"] = bool(quote) and normalize(quote) in normalize(transcript_text)
    return answers


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


THEMES_SYSTEM_PROMPT = """You are analyzing 3 expert-call transcripts from the same \
project. Identify (a) themes/points at least two experts agree on, and (b) points \
where experts disagree or contradict each other. Only use what is explicitly stated. \
For every theme or disagreement, cite which expert(s) said it and include a short \
verbatim quote with timestamp for each. Respond ONLY with JSON:
{"common_themes": [{"theme": "...", "experts": ["Expert 1", "Expert 2"], \
"evidence": [{"expert": "...", "quote": "...", "timestamp": "..."}]}],
"disagreements": [{"topic": "...", "evidence": [{"expert": "...", "quote": "...", \
"timestamp": "..."}]}]}"""


def analyze_themes(client, expert_texts: dict):
    combined = "\n\n".join(
        f"=== {name} ===\n{text}" for name, text in expert_texts.items()
    )
    raw = call_llm(client, THEMES_SYSTEM_PROMPT, combined)
    return json.loads(raw)


QA_SYSTEM_PROMPT = """You answer questions using ONLY the 3 expert transcripts \
provided. If the answer isn't in the transcripts, say so plainly. Always cite \
which expert and timestamp the answer came from, and include a short verbatim quote. \
Respond ONLY with JSON:
{"answer": "...", "citations": [{"expert": "...", "timestamp": "...", "quote": "..."}]}"""


def ask_question(client, expert_texts: dict, question: str):
    combined = "\n\n".join(
        f"=== {name} ===\n{text}" for name, text in expert_texts.items()
    )
    raw = call_llm(client, QA_SYSTEM_PROMPT, f"{combined}\n\nQUESTION: {question}")
    return json.loads(raw)


# ---------------------------------------------------------------------------
# 3. Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Expert Transcript Analyzer", layout="wide")
st.title("Expert Call Transcript Analyzer")

with st.sidebar:
    st.subheader("Setup")
    try:
        default_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        default_key = ""
    api_key = st.text_input(
        "Groq API key",
        value=default_key,
        type="password",
        help="console.groq.com/keys — free tier. Pre-filled automatically if set in Streamlit secrets.",
    )
    st.caption("Supports `[00:01:23] Speaker: text` or a timestamp on its own\nline followed by `Speaker: text` on the next.")

questions_raw = st.text_area(
    "Interview-guide questions (one per line)",
    height=150,
    value=(
        "How would you describe current adoption of robotic surgery in your market?\n"
        "What are the main barriers to adoption?\n"
        "How important are hospital budgets and ROI in purchasing decisions?\n"
        "How important are surgeon training and clinical outcomes?\n"
        "What adoption trend do you expect over the next 3-5 years?\n"
        "What is the typical hospital decision-making timeline for purchasing a new robotic system?"
    ),
)
questions = [q.strip() for q in questions_raw.splitlines() if q.strip()]

st.subheader("Upload the 3 transcripts")
cols = st.columns(3)
uploads = {}
for i, col in enumerate(cols, start=1):
    with col:
        f = col.file_uploader(f"Expert {i}", type=["txt"], key=f"upl_{i}")
        if f:
            uploads[f"Expert {i}"] = f.read().decode("utf-8")

run = st.button("Analyze", type="primary", disabled=not (api_key and len(uploads) == 3 and questions))

if "results" not in st.session_state:
    st.session_state.results = None

if run:
    client = Groq(api_key=api_key)
    parsed = {name: parse_transcript(text) for name, text in uploads.items()}
    headers = {name: header for name, (header, _segs) in parsed.items()}
    segs_by_name = {name: segs for name, (_header, segs) in parsed.items()}

    empty = [name for name, segs in segs_by_name.items() if not segs]
    if empty:
        st.error(
            f"Couldn't find any timestamped lines in: {', '.join(empty)}. "
            "Check the transcript uses `[HH:MM:SS] Speaker: text` or a timestamp "
            "on its own line followed by `Speaker: text`."
        )
    else:
        with st.spinner("Answering interview-guide questions per expert..."):
            per_expert_answers = {
                name: extract_answers(client, segs, questions) for name, segs in segs_by_name.items()
            }
        with st.spinner("Finding cross-expert themes and disagreements..."):
            expert_texts = {name: transcript_as_numbered_text(segs) for name, segs in segs_by_name.items()}
            themes = analyze_themes(client, expert_texts)
        st.session_state.results = {
            "answers": per_expert_answers,
            "themes": themes,
            "expert_texts": expert_texts,
            "headers": headers,
        }

results = st.session_state.results

if results:
    tab_names = list(results["answers"].keys()) + ["Cross-Expert Analysis", "Ask a Question"]
    tabs = st.tabs(tab_names)

    for tab, name in zip(tabs, results["answers"].keys()):
        with tab:
            if results["headers"].get(name):
                st.caption(results["headers"][name].replace("\n", " · "))
            for a in results["answers"][name]:
                verified_badge = "✅ verified" if a["verified"] else ("⚠️ unverified" if a["quote"] else "—")
                st.markdown(f"**Q: {a['question']}**")
                st.write(a["answer"])
                if a["quote"]:
                    st.caption(f"[{a['timestamp']}] \"{a['quote']}\" — {verified_badge}")
                st.divider()

    with tabs[-2]:
        st.markdown("### Common themes")
        for t in results["themes"].get("common_themes", []):
            st.markdown(f"**{t['theme']}** — agreed by: {', '.join(t['experts'])}")
            for e in t.get("evidence", []):
                st.caption(f"{e['expert']} [{e['timestamp']}]: \"{e['quote']}\"")
            st.divider()
        st.markdown("### Disagreements")
        for d in results["themes"].get("disagreements", []):
            st.markdown(f"**{d['topic']}**")
            for e in d.get("evidence", []):
                st.caption(f"{e['expert']} [{e['timestamp']}]: \"{e['quote']}\"")
            st.divider()

    with tabs[-1]:
        q = st.text_input("Ask something across all 3 transcripts")
        if st.button("Ask"):
            client = Groq(api_key=api_key)
            answer = ask_question(client, results["expert_texts"], q)
            st.write(answer["answer"])
            for c in answer.get("citations", []):
                st.caption(f"{c['expert']} [{c['timestamp']}]: \"{c['quote']}\"")
else:
    st.info("Enter your Groq API key, upload 3 transcripts, and click Analyze.")