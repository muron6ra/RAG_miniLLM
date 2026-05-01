import os
import sys
import json
import logging
import argparse
import re
from pathlib import Path

import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer

DATA_DIR = Path("./data")
INDEX_DIR = Path("./index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual, supports Turkish
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"    # Ultra-fast, decent Turkish support

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 4
MAX_NEW_TOKENS = 256

CONVERSATIONAL_KEYWORDS: set[str] = {
    "merhaba", "selam", "hey", "hi", "hello", "günaydın", "iyi günler",
    "iyi akşamlar", "iyi geceler", "nasılsın", "naber", "ne haber",
    "keyifler nasıl", "ne biliyorsun", "neler biliyorsun", "ne bilirsin",
    "kimsin", "sen kimsin", "adın ne", "adin ne", "ne yapabilirsin",
    "ne yaparsın", "yardım", "nasıl kullanırım", "teşekkür", "teşekkürler",
    "sağ ol", "eyvallah", "güle güle", "görüşürüz", "hoşça kal",
    "bay bay", "görüşmek üzere", "aferin", "bravo", "harika", "nereden",
    "iyi misin", "yakışıyor mu", "şeklın ne", "ne düşünüyorsun",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_documents(data_dir: Path) -> list[dict]:
    """Read all .txt files from data_dir. Returns list of {source, text} dicts."""
    docs = []
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        log.error(f"No .txt files found in '{data_dir}'. Add your documents and retry.")
        sys.exit(1)

    for path in txt_files:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"source": path.name, "text": text})
            log.info(f"Loaded: {path.name} ({len(text):,} chars)")

    log.info(f"Loaded {len(docs)} document(s).")
    return docs


def split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter — splits on '. ', '? ', '! '."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(doc: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split a document into overlapping chunks respecting sentence boundaries.
    Returns list of {source, text, chunk_id} dicts.
    """
    sentences = split_into_sentences(doc["text"])
    chunks = []
    current_chunk = ""
    chunk_id = 0

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append({
                "source": doc["source"],
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
            })
            chunk_id += 1

            current_chunk = current_chunk[-overlap:] + " " + sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence

    if current_chunk.strip():
        chunks.append({
            "source": doc["source"],
            "chunk_id": chunk_id,
            "text": current_chunk.strip(),
        })

    return chunks


def build_chunks(docs: list[dict]) -> list[dict]:
    """Chunk all documents and return a flat list."""
    all_chunks = []
    for doc in docs:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        log.info(f"  {doc['source']}: {len(doc_chunks)} chunk(s)")
    log.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks



def load_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformer embedding model."""
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def build_faiss_index(chunks: list[dict], embed_model: SentenceTransformer) -> faiss.Index:
    """Generate embeddings for all chunks and build a FAISS index."""
    texts = [c["text"] for c in chunks]
    log.info("Generating embeddings (this may take a moment)...")
    embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    log.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(index: faiss.Index, chunks: list[dict]) -> None:
    """Persist the FAISS index and chunk metadata to disk."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    log.info(f"Index saved → {FAISS_INDEX_PATH}")


def load_index() -> tuple[faiss.Index, list[dict]]:
    """Load a persisted FAISS index and chunk metadata."""
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    log.info(f"Index loaded: {index.ntotal} vectors from disk.")
    return index, chunks


def index_exists() -> bool:
    return FAISS_INDEX_PATH.exists() and CHUNKS_PATH.exists()


def retrieve(query: str, index: faiss.Index, chunks: list[dict],
             embed_model: SentenceTransformer, top_k: int = TOP_K) -> list[dict]:
    """Embed the query and return the top-k most relevant chunks."""
    query_vec = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


def load_llm() -> pipeline:
    """Load the local LLM as a text-generation pipeline."""
    log.info(f"Loading LLM: {LLM_MODEL}  (first run will download the model)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    log.info(f"Running on: {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    pipe_device = 0 if device == "cuda" else -1

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.1,
        device=pipe_device,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_pipeline.tokenizer = tokenizer 
    return gen_pipeline


def build_messages(query: str, context_chunks: list[dict]) -> list[dict]:
    """Build a list of messages for the RAG chat template."""
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    messages = [
        {"role": "system", "content": "Sen yardımcı bir asistansın. Soruyu SADECE bağlamı kullanarak, kısa ve öz bir şekilde cevapla. Eğer cevap bağlamda yoksa, tam olarak 'Bilmiyorum.' de."},
        {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {query}"}
    ]
    return messages


def generate_answer(query: str, context_chunks: list[dict], gen_pipeline) -> str:
    """Generate an answer given a query and relevant chunks."""
    if not context_chunks:
        answer = "Malesef aradığınız bilgi elimde yok."
        print(f"\nAsistan: {answer}")
        return answer

    context = "\n\n---\n\n".join(
        f"{c['text']}" for c in context_chunks
    )
    prompt = f"{context}\n\nSoru: {query}\nCevap:"

    streamer = TextStreamer(gen_pipeline.tokenizer, skip_prompt=True, skip_special_tokens=True)
    print("\nAsistan: ", end="", flush=True)
    outputs = gen_pipeline(prompt, streamer=streamer)
    full_text: str = outputs[0]["generated_text"]
    answer = full_text[len(prompt):].strip()
    return answer if answer else "Bilmiyorum."

def is_conversational(query: str) -> bool:
    """Sorgunun sohbet nitelikli olup olmadığını tespit eder."""
    q = query.lower().strip()
    return any(kw in q for kw in CONVERSATIONAL_KEYWORDS)


def generate_chat_response(query: str, gen_pipeline) -> str:
    """Belge bağlamı olmaksızın sohbet sorularına LLM ile yanıt üretir."""
    prompt = f"{query}\nCevap:"
    streamer = TextStreamer(gen_pipeline.tokenizer, skip_prompt=True, skip_special_tokens=True)
    print("\nAsistan: ", end="", flush=True)
    outputs = gen_pipeline(prompt, streamer=streamer)
    full_text: str = outputs[0]["generated_text"]
    answer = full_text[len(prompt):].strip()
    return answer if answer else "..."


def chat_loop(index: faiss.Index, chunks: list[dict],
              embed_model: SentenceTransformer, gen_pipeline) -> None:
    """Interactive terminal chat loop."""
    print("\n" + "=" * 60)
    print("  Kişisel RAG — Yerel Bilgi Asistanı")
    print("  Sorunuzu yazın. Çıkmak için 'exit' veya 'quit' yazın.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Görüşmek üzere! 👋")
            break

        if is_conversational(query):
            generate_chat_response(query, gen_pipeline)
            print("\n" + "-" * 60)
            continue

        relevant_chunks = retrieve(query, index, chunks, embed_model)
        log.debug(f"Retrieved {len(relevant_chunks)} chunks.")

        generate_answer(query, relevant_chunks, gen_pipeline)
        
        sources = list({c["source"] for c in relevant_chunks})
        print(f"\n\n  [Kaynaklar: {', '.join(sources)}]\n")
        print("-" * 60)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Personal RAG — Local LLM + FAISS")
    parser.add_argument("--data", type=Path, default=DATA_DIR,
                        help="Path to folder containing .txt files (default: ./data)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild the FAISS index even if it already exists")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        log.info(f"CPU threads optimized: {num_threads}")

    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    global DATA_DIR
    DATA_DIR = args.data

    embed_model = load_embedding_model()

    if index_exists() and not args.rebuild:
        log.info("Existing index found. Loading from disk (use --rebuild to regenerate).")
        index, chunks = load_index()
    else:
        if args.rebuild:
            log.info("--rebuild flag set. Rebuilding index...")
        else:
            log.info("No index found. Building from scratch...")

        docs = load_documents(DATA_DIR)
        chunks = build_chunks(docs)
        index = build_faiss_index(chunks, embed_model)
        save_index(index, chunks)

    gen_pipeline = load_llm()

    chat_loop(index, chunks, embed_model, gen_pipeline)


if __name__ == "__main__":
    main()
