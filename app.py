import os
import json
import uuid
import re
import io
from typing import List, Dict, Any, Tuple

import streamlit as st
from openai import OpenAI

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


# ---------------------------
# Utilities
# ---------------------------

def ensure_api_key():
    if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key.strip()


def get_openai_client():
    # Requirement: Create the client using: client = OpenAI()
    client = OpenAI()
    return client


def read_pdf(file) -> str:
    if PyPDF2 is None:
        return ""
    try:
        reader = PyPDF2.PdfReader(file)
        texts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            texts.append(txt)
        return "\n".join(texts)
    except Exception:
        # If PyPDF2 fails, try reading bytes and re-initializing
        try:
            data = file.read()
            file.seek(0)
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            texts = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                texts.append(txt)
            return "\n".join(texts)
        except Exception:
            return ""


def extract_text_from_files(files: List[Any]) -> str:
    all_texts = []
    for f in files:
        text = read_pdf(f)
        if text:
            all_texts.append(text)
    return "\n\n".join(all_texts)


def clean_text(text: str) -> str:
    # Normalize whitespace
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 6000, overlap: int = 400) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current += (("\n\n" if current else "") + para)
        else:
            if current:
                chunks.append(current)
            # If single paragraph too large, hard-split
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    chunks.append(para[start:end])
                    start = end - overlap if end < len(para) else end
            else:
                current = para
                continue
            current = ""
    if current:
        chunks.append(current)

    # Add simple overlaps between chunks to preserve context
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
            else:
                prev = chunks[i - 1]
                tail = prev[-overlap:]
                overlapped.append(tail + "\n\n" + ch)
        return overlapped
    return chunks


def build_messages(chunk: str, card_style: str, format_hint: str, max_cards: int, deck_name: str, additional_instr: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a helpful assistant that converts study notes into high-quality, testable flashcards. "
        "Rules:\n"
        "- Use only the provided text; do not add external facts.\n"
        "- Write clearly and concisely; avoid ambiguity.\n"
        "- Prefer atomic facts/concepts per card.\n"
        "- If content is insufficient for some cards, return fewer cards.\n"
        "- Output must be strictly valid JSON matching the schema.\n"
    )
    schema = {
        "cards": [
            {
                "id": "uuid4 string",
                "front": "string",
                "back": "string",
                "tags": ["optional", "tags"]
            }
        ],
        "deck": deck_name
    }

    user_prompt = f"""
Create up to {max_cards} high-quality flashcards from the following study notes CHUNK, adhering to the requested style and format hints.

Style: {card_style}
Format hints: {format_hint if format_hint else "N/A"}
Deck name: {deck_name if deck_name else "My Deck"}
Additional instructions: {additional_instr if additional_instr else "N/A"}

Constraints:
- Only use information from the CHUNK below.
- Ensure each card is self-contained.
- Keep front concise; put full detail on the back.
- Prefer terminology consistent with the notes.

Return strictly valid JSON with keys: "deck" (string) and "cards" (array of objects with keys "id", "front", "back", and optional "tags" (array of strings)).
Do not include any extra text before or after the JSON.

JSON schema example:
{json.dumps(schema, indent=2)}

CHUNK:
\"\"\"{chunk}\"\"\"
"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_json_cards(raw: str) -> Tuple[str, List[Dict[str, Any]]]:
    # Strip code fences if present
    raw = raw.strip()
    fence_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(0)
    try:
        data = json.loads(raw)
        deck = data.get("deck", "My Deck")
        cards = data.get("cards", [])
        normalized = []
        for c in cards:
            card = {
                "id": c.get("id") or str(uuid.uuid4()),
                "front": (c.get("front") or "").strip(),
                "back": (c.get("back") or "").strip(),
                "tags": c.get("tags") or []
            }
            if card["front"] and card["back"]:
                normalized.append(card)
        return deck, normalized
    except Exception:
        return "My Deck", []


def dedupe_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for c in cards:
        key = (c["front"].strip().lower(), c["back"].strip().lower())
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


def cards_to_csv(cards: List[Dict[str, Any]]) -> str:
    # Simple CSV with Front,Back,Tags
    # Escape quotes and commas
    lines = ["Front,Back,Tags"]
    for c in cards:
        front = '"' + c["front"].replace('"', '""') + '"'
        back = '"' + c["back"].replace('"', '""') + '"'
        tags = '"' + ", ".join(c.get("tags", [])) .replace('"', '""') + '"'
        lines.append(",".join([front, back, tags]))
    return "\n".join(lines)


def cards_to_tsv_for_anki(cards: List[Dict[str, Any]]) -> str:
    # Anki basic: front<TAB>back<TAB>tags
    lines = []
    for c in cards:
        tags = " ".join(c.get("tags", []))
        lines.append(f"{c['front']}\t{c['back']}\t{tags}")
    return "\n".join(lines)


def generate_cards_from_chunks(chunks: List[str], model: str, temperature: float, per_chunk_limit: int,
                               card_style: str, format_hint: str, deck_name: str, additional_instr: str) -> List[Dict[str, Any]]:
    all_cards = []
    ensure_api_key()
    client = get_openai_client()
    for idx, ch in enumerate(chunks, start=1):
        messages = build_messages(ch, card_style, format_hint, per_chunk_limit, deck_name, additional_instr)
        with st.spinner(f"Generating flashcards from chunk {idx}/{len(chunks)}..."):
            try:
                response = client.chat.completions.create(
                    model=model,  # or "gpt-4"
                    messages=messages,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                _, cards = parse_json_cards(content)
                all_cards.extend(cards)
            except Exception as e:
                st.warning(f"OpenAI API error on chunk {idx}: {e}")
    return all_cards


# ---------------------------
# Streamlit App
# ---------------------------

def init_session():
    defaults = {
        "cards": [],
        "deck_name": "My Deck",
        "study_index": 0,
        "show_answer": False,
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sidebar_controls():
    st.sidebar.header("Settings")
    st.sidebar.text_input("OpenAI API Key", type="password", key="openai_api_key", help="Stored only in session memory.")
    model = st.sidebar.selectbox("Model", options=["gpt-4", "gpt-3.5-turbo"], index=0)
    temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.5, 0.3, 0.1)
    max_cards = st.sidebar.number_input("Total cards (approx)", min_value=1, max_value=200, value=30, step=1)
    per_chunk = st.sidebar.number_input("Max cards per chunk", min_value=1, max_value=50, value=10, step=1)
    return model, temperature, int(max_cards), int(per_chunk)


def main():
    st.set_page_config(page_title="PDF to Flashcards", page_icon="🃏", layout="wide")
    init_session()

    st.title("🃏 PDF Notes → Flashcards Generator")
    st.write("Upload your notes PDFs, generate flashcards, and study them interactively.")

    model, temperature, max_cards_total, per_chunk_limit = sidebar_controls()

    uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
    deck_name = st.text_input("Deck Name", st.session_state.get("deck_name", "My Deck"))
    card_style = st.selectbox("Flashcard style", options=[
        "Q&A (question on front, answer on back)",
        "Cloze deletion (fill-in-the-blank)",
        "Term → Definition",
        "Concept → Explanation",
        "Process → Steps"
    ], index=0)
    format_hint = st.text_input("Formatting hints (optional)", placeholder="e.g., Keep fronts <= 20 words; Use bold for keywords")
    additional_instr = st.text_area("Additional instructions (optional)", placeholder="Any specific curriculum, exam focus, or teaching style to emulate?")
    st.session_state["deck_name"] = deck_name

    colA, colB = st.columns([1, 1])
    with colA:
        generate = st.button("Generate Flashcards", type="primary", use_container_width=True)
    with colB:
        clear = st.button("Clear Deck", use_container_width=True)

    if clear:
        st.session_state.cards = []
        st.session_state.study_index = 0
        st.session_state.show_answer = False
        st.experimental_rerun()

    if generate:
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            raw_text = extract_text_from_files(uploaded_files)
            raw_text = clean_text(raw_text)
            if not raw_text:
                st.error("Could not extract text from the uploaded PDFs. Please ensure the files contain selectable text.")
            else:
                chunks = chunk_text(raw_text, chunk_size=6000, overlap=400)
                approx_chunks_needed = max(1, min(len(chunks), (max_cards_total + per_chunk_limit - 1) // per_chunk_limit))
                chunks = chunks[:approx_chunks_needed]

                cards = generate_cards_from_chunks(
                    chunks=chunks,
                    model=model,
                    temperature=temperature,
                    per_chunk_limit=per_chunk_limit,
                    card_style=card_style,
                    format_hint=format_hint,
                    deck_name=deck_name,
                    additional_instr=additional_instr
                )

                cards = dedupe_cards(cards)
                # Trim to total desired
                cards = cards[:max_cards_total]
                st.session_state.cards = cards
                st.session_state.study_index = 0
                st.session_state.show_answer = False

    st.divider()
    st.subheader("Deck")
    count = len(st.session_state.cards)
    st.caption(f"Deck: {deck_name} • {count} cards")

    if count > 0:
        # Download options
        col1, col2, col3 = st.columns(3)
        with col1:
            json_data = json.dumps({"deck": deck_name, "cards": st.session_state.cards}, indent=2, ensure_ascii=False)
            st.download_button("Download JSON", data=json_data, file_name=f"{deck_name}.json", mime="application/json", use_container_width=True)
        with col2:
            csv_data = cards_to_csv(st.session_state.cards)
            st.download_button("Download CSV", data=csv_data, file_name=f"{deck_name}.csv", mime="text/csv", use_container_width=True)
        with col3:
            tsv_data = cards_to_tsv_for_anki(st.session_state.cards)
            st.download_button("Download Anki TSV", data=tsv_data, file_name=f"{deck_name}.tsv", mime="text/tab-separated-values", use_container_width=True)

        st.markdown("")

        # Study Mode
        st.subheader("Study Mode")
        idx = st.session_state.study_index
        card = st.session_state.cards[idx]

        st.write(f"Card {idx + 1} of {count}")
        st.text_area("Front", card["front"], height=150)
        if st.session_state.show_answer:
            st.text_area("Back", card["back"], height=200)
            if card.get("tags"):
                st.caption("Tags: " + ", ".join(card["tags"]))
        else:
            st.info("Click 'Flip' to reveal the answer.")

        col_prev, col_flip, col_next = st.columns([1, 1, 1])
        with col_prev:
            if st.button("Prev", use_container_width=True):
                st.session_state.study_index = (st.session_state.study_index - 1) % count
                st.session_state.show_answer = False
                st.experimental_rerun()
        with col_flip:
            if st.button("Flip", use_container_width=True):
                st.session_state.show_answer = not st.session_state.show_answer
        with col_next:
            if st.button("Next", use_container_width=True):
                st.session_state.study_index = (st.session_state.study_index + 1) % count
                st.session_state.show_answer = False
                st.experimental_rerun()

        with st.expander("Preview all cards"):
            for i, c in enumerate(st.session_state.cards, start=1):
                st.markdown(f"**{i}. {c['front']}**")
                st.write(c["back"])
                if c.get("tags"):
                    st.caption("Tags: " + ", ".join(c["tags"]))
                st.markdown("---")
    else:
        st.info("No cards yet. Upload PDFs and click 'Generate Flashcards'.")


if __name__ == "__main__":
    main()