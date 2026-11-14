"""Local RAG pipeline using LocalTfidfEmbedder, FAISS, BPETokenizer, and MiniLLM."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import torch

from embedder import LocalTfidfEmbedder
from model import MiniLLM
from tokenizer import BPETokenizer, format_chat_prompt, format_rag_prompt

_DEFAULT_INDEX_DIR = Path("./index")

DATA_DIR = Path("./data")
INDEX_DIR = _DEFAULT_INDEX_DIR
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMBEDDER_PATH = INDEX_DIR / "embedder.json"

CHECKPOINT_DIR = Path("./checkpoints")
MODEL_PATH = CHECKPOINT_DIR / "mini_llm.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"

CHUNK_SIZE = 750
CHUNK_OVERLAP = 80
MIN_PARA_MERGE_LEN = 120
TOP_K = 2
MIN_SCORE = 0.05
MIN_RATIO = 0.55
MAX_NEW_TOKENS = 128

CONVERSATIONAL_KEYWORDS: set[str] = {
    "merhaba", "merhabalar", "selam", "selamlar", "selamün aleyküm",
    "hey", "hi", "hello", "günaydın", "gunaydin", "günaydınlar",
    "tünaydın", "tunaydin", "hayırlı sabahlar", "iyi günler", "iyi öğlenler",
    "iyi akşamlar", "iyi aksamlar", "hayırlı akşamlar", "iyi geceler",
    "hayırlı geceler", "hoş bulduk", "hos bulduk", "mrb", "slm", "kimse var mı",
    "nasılsın", "nasilsin", "nasılsınız", "naber", "nbr", "ne haber",
    "keyifler nasıl", "keyfin nasıl", "durumlar nasıl", "iyi misin",
    "neler yapıyorsun", "ne yapıyorsun", "günün nasıl geçiyor",
    "nasıl gidiyor", "ne var ne yok", "keyfin yerinde mi", "her şey yolunda mı",
    "nasıl hissediyorsun", "ne biliyorsun", "neler biliyorsun", "ne bilirsin",
    "kimsin", "sen kimsin", "kimsin sen", "sen nesin", "hangi asistansın",
    "adın ne", "adin ne", "adın nedir", "adin nedir", "ismin ne", "ismin nedir",
    "kendini tanıtır mısın", "kendinden bahseder misin", "görevin nedir",
    "ne yapabilirsin", "ne yaparsın", "neler yapabilirsin", "neler yapabiliyorsun",
    "neler yaparsın", "ne yapabiliyorsun", "yapabildiklerin", "yeteneklerin",
    "yeteneklerin neler", "nasıl çalışırsın", "nasıl çalışıyorsun",
    "ne işe yararsın", "yardım", "yardım eder misin", "yardımcı olur musun",
    "yardımına ihtiyacım var", "bana yardım et",
    "nasıl kullanırım", "seni nasıl kullanabilirim", "nereden çalışıyorsun",
    "bulutta mı çalışıyorsun", "ücretsiz misin", "hazır bir model misin",
    "chatgpt misin", "insan mısın robot musun",
    "seni kim yaptı", "seni kim geliştirdi", "internete bağlı mısın",
    "internetin var mı", "açık kaynak mısın", "komutların var mı",
    "cevapların neden kısa", "yanlış cevap verirsen", "özet çıkarabilir misin",
    "verilerim", "verilerimi", "bilgilerim güvende", "gizliliğim",
    "konuşmalarımız gizli", "güvende mi",
    "teşekkür", "teşekkürler", "tesekkurler", "teşekkür ederim",
    "tesekkur ederim", "teşekkür ediyorum", "tşk",
    "sağ ol", "sağol", "sağ olasın", "eyvallah", "eyw", "rica ederim",
    "çok yardımcı oldun", "sevdim seni",
    "güle güle", "görüşürüz", "gorusuruz", "hoşça kal", "hoşça kalın",
    "hoşçakal", "hoşçakalın", "bay bay", "bye", "görüşmek üzere",
    "kendine iyi bak", "kendinize iyi bakın", "kendine dikkat et",
    "sonra konuşuruz", "yarın görüşürüz",
    "iyi çalışmalar", "kolay gelsin", "mutlu günler",
    "hayırlı günler", "çıkıyorum", "şimdi gidiyorum", "ben kaçtım",
    "aferin", "bravo", "harika", "harikasın", "süper", "süpersin",
    "çok iyi", "çok iyisin", "mükemmel", "eline sağlık", "işime yaradı",
    "çok başarılısın",
    "saat kaç", "saat kac", "saat kaçtır", "saati söyleyebilir misin",
    "hava durumu", "bugün hava nasıl", "bugun hava nasil", "hava nasıl",
    "yağmur yağacak mı", "bugün ayın kaçı", "bugün günlerden ne",
    "bugün tarih", "bugünkü tarih", "bugünün tarihi", "tarihi söyleyebilir misin",
    "hangi gündeyiz", "hangi tarihteyiz",
    "dolar kuru", "dolar kaç tl", "euro ne kadar", "haberlerde ne var",
    "güncel haberler", "maç skoru",
    "haftanın günleri", "haftada kaç gün", "bir haftada kaç gün",
    "hafta sonu", "hafta içi", "mevsimler nelerdir", "kaç mevsim",
    "dört mevsim", "kış ayları", "yaz ayları", "ilkbahar ayları",
    "sonbahar ayları", "yılın ayları", "kaç ay var", "bir senede kaç ay",
    "yılın ilk ayı", "yılın son ayı", "bir günde kaç saat",
    "bir saatte kaç dakika", "bir dakikada kaç saniye", "bir yılda kaç gün",
    "artık yıl", "bir hafta kaç saat", "şubat ayı kaç gün",
    "türkiye başkenti", "türkiye'nin başkenti", "türkiyenin başkenti",
    "turkiyenin baskenti", "ülkemizin başkenti", "başkentimiz",
    "türkiye'nin en kalabalık", "türkiye'nin en büyük şehri",
    "türk bayrağının renkleri", "türk bayrağı", "istiklal marşı",
    "cumhuriyet ne zaman", "cumhuriyet hangi tarihte",
    "kaç derecede kaynar", "kaç derecede donar", "kaynama noktası",
    "donma noktası", "suyun formülü", "suyun kimyasal formülü", "h2o",
    "güneş sisteminde", "kaç gezegen var", "gezegen sayısı", "gezegenler nelerdir",
    "dünyanın şekli", "dünyanın uydusu", "dünyanın doğal uydusu",
    "en yakın yıldız", "en büyük organ", "ana renkler",
    "gökkuşağında kaç renk", "üçgenin iç açıları", "karenin kaç kenarı",
    "bir düzine", "bir deste", "kilometrede kaç metre", "1 kilometre kaç metre",
    "odaklanmak", "odaklanamıyorum", "dikkat dağınıklığı", "kahve mi çay mı",
    "kitap okumanın", "neden kitap okumalıyız", "düzenli uyku",
    "uyku düzeni", "su içmenin önemi", "ne kadar su içmeliyiz",
    "neden düzenli uyumalıyız", "neden su içmeliyiz",
    "motivasyon", "motivasyonum", "yorgunum", "sınav öncesi", "not almak",
    "stresli hissettiğimde", "stresli olduğumda", "düzenli yürüyüş",
    "dinç uyanmak", "şaka", "saka", "espri", "kısa bir şaka",
    "komik bir şey", "beni güldürür müsün", "bilmece", "boş zamanlarımda",
    "yeni bir şey öğrenirken",
    "yakışıyor mu", "şeklın ne", "ne düşünüyorsun",
}

# Exact standalone short/filler queries routed to chat without risking substring hits.
_EXACT_CONVERSATIONAL_QUERIES: set[str] = {
    "sa", "...", "???", "asdf", "a", "hmm", "şey",
}

# Compiled regex patterns handling Turkish suffixes, filler words, and word-order variants.
_CONVERSATIONAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"(?<!\w)te[sş]ekk[uü]r\w*",
        r"(?<!\w)g[oö]r[uü][sş][uü]r[uü]z(?!\w)",
        r"(?<!\w)ho[sş][cç]a\s*kal(?:[ıi]n)?(?!\w)",
        r"(?<!\w)g[uü]nayd[ıi]n(?:lar)?(?!\w)",
        r"(?<!\w)nas[ıi]ls[ıi]n(?:[ıi]z)?(?!\w)",
        r"(?<!\w)kimsin(?:iz)?(?!\w)",
        r"(?<!\w)(?:ad[ıi]n|ismin)\s+ne(?:dir)?(?!\w)",
        r"(?<!\w)neler\s+yap(?:abilirsin|abiliyorsun|abiliyorsunuz|ars[ıi]n|[ıi]yorsun)(?:[ıi]z|uz)?(?!\w)",
        r"(?<!\w)ne\s+(?:gibi\s+i[sş]ler\s+)?yap(?:abilirsin|abiliyorsun|abiliyorsunuz|ars[ıi]n|[ıi]yorsun)(?:[ıi]z|uz)?(?!\w)",
        r"(?<!\w)(?:hangi\s+i[sş]leri\s+yap(?:abilirsin|abiliyorsun|ars[ıi]n|[ıi]yorsun)\w*|yapabildiklerin\w*|yapabilece[gğ]in\s+[sş]eyler\w*|yeteneklerin\w*)(?!\w)",
        r"(?<!\w)(?:hangi\s+konularda|ne\s+konuda)\s+yard[ıi]mc[ıi]\s+ol\w*",
        r"(?<!\w)(?:yard[ıi]mc[ıi]\s+ol(?:ur|abilir)\s+m[ıi]s[ıi]n(?:[ıi]z)?|bana\s+yard[ıi]m\s+et)(?!\w)",
        r"(?<!\w)(?:hangi\s+dili\s+biliyorsun|t[uü]rk[cç]e\s+biliyor\s+musun|ingilizce\s+anl[ıi]yor\s+musun)(?!\w)",
        r"(?<!\w)(?:bulutta\s+m[ıi]\s+[cç]al[ıi][sş][ıi]yorsun|[uü]cretsiz\s+misin|haz[ıi]r\s+bir\s+model\s+misin|chatgpt\s+misin|insan\s+m[ıi]s[ıi]n\s+robot\s+musun)(?!\w)",
        r"(?<!\w)(?:nas[ıi]l\s+e[gğ]itil(?:din|irsin)|hangi\s+verilerle\s+e[gğ]itildin|seni\s+nas[ıi]l\s+e[gğ]itebilirim)(?!\w)",
        r"(?<!\w)(?:(?:bir\s+[sş]ey\s+)?sorabilir\s+miyim|bir\s+sorum\s+var|belge\s+ekleyebilir\s+miyim|kendi\s+dosyalar[ıi]m[ıi]\s+nas[ıi]l\s+eklerim|hangi\s+dosya\s+t[uü]rlerini\s+okursun|belgelerde\s+olmayan\s+bir\s+[sş]ey\s+sorarsam|nas[ıi]l\s+daha\s+iyi\s+soru\s+sorar[ıi]m|sohbeti\s+nas[ıi]l\s+bitiririm|[cç][ıi]k[ıi][sş]\s+nas[ıi]l\s+yap[ıi]l[ıi]r|kod\s+yazabilir\s+misin|[cç]eviri\s+yapabilir\s+misin)(?!\w)",
        r"(?<!\w)(?:onu\s+s[oö]yle|bunu\s+bil\s+bakal[ıi]m|akl[ıi]mdan\s+ne\s+ge[cç]iyor|yar[ıi]n\s+ne\s+olacak|[sş]ifremi\s+unuttum\w*|ne\s+demek\s+istedi[gğ]imi\s+anlad[ıi]n\s+m[ıi])(?!\w)",
        r"(?<!\w)(?:verilerim\w*|bilgilerim\w*|gizlili[gğ]im\w*|konu[sş]malar[ıi]m[ıi]z\w*)(?!\w)",
        r"(?<!\w)g[uü]vende\s+mi(?:yim|yiz)?(?!\w)",
        r"(?<!\w)saat\s+ka[cç](?:t[ıi]r)?(?!\w)",
        r"(?<!\w)hava\s+durumu\b",
        r"(?<!\w)hava\s+(?:yar[ıi]n\s+)?nas[ıi]l\b",
        r"(?<!\w)bug[uü]n(?:k[uü]|n[uü]n)?\s+(?:hava\b|tarih\w*)",
        r"(?<!\w)(?:tarih\s+ka[cç]|tarihi\s+s[oö]yleyebilir\s+misin|hangi\s+tarihteyiz|hangi\s+g[uü]ndeyiz)(?!\w)",
        r"(?<!\w)(?:(?:bir|1)\s+)?hafta(?:da)?\s+ka[cç]\s+(?:g[uü]n|saat)\w*",
        r"(?<!\w)(?:(?:bir|1)\s+)?(?:y[ıi]l(?:da)?|sene(?:de)?)\s+ka[cç]\s+(?:ay|g[uü]n|mevsim)\w*",
        r"(?<!\w)(?:(?:bir|1)\s+)?(?:y[ıi]l|sene)\s+ka[cç]\s+ayd[ıi]r(?!\w)",
        r"(?<!\w)(?:bir|1)\s+(?:g[uü]n(?:de)?\s+ka[cç]\s+saat|saat(?:te)?\s+ka[cç]\s+dakika|dakika(?:da)?\s+ka[cç]\s+saniye|kilometre(?:de)?\s+ka[cç]\s+metre)\w*",
        r"(?<!\w)[sş]ubat\s+ay[ıi]\s+(?:normalde\s+)?ka[cç]\s+g[uü]n\b",
        r"(?<!\w)(?:d[oö]rt\s+mevsim\w*|ka[cç]\s+mevsim\w*)(?!\w)",
        r"(?<!\w)(?:t[uü]rkiye(?:'?nin)?|t[uü]rkiyenin|[uü]lkemizin)\s+ba[sş]kent\w*",
        r"(?<!\w)ba[sş]kentimiz\w*",
        r"(?<!\w)t[uü]rkiye(?:'?nin)?\s+en\s+(?:kalabal[ıi]k|b[uü]y[uü]k)\s+(?:[sş]ehri|ili)\b",
        r"(?<!\w)(?:t[uü]rk\s+bayra[gğ]\w*|istiklal\s+mar[sş]\w*|cumhuriyet\s+(?:ne\s+zaman|hangi\s+tarihte))\b",
        r"(?<!\w)ka[cç]\s+(?:santigrat\s+)?derece(?:de)?\s+(?:kaynar|donar)(?!\w)",
        r"(?<!\w)(?:saf\s+)?su(?:yun)?\s+(?:kaynama\s+(?:noktas[ıi]|s[ıi]cakl[ıi][gğ][ıi])|donma\s+(?:noktas[ıi]|s[ıi]cakl[ıi][gğ][ıi])|ka[cç]\s+derece\s+so[gğ]ukta\s+donar|ne\s+zaman\s+kaynar|(?:kimyasal\s+)?form[uü]l[uü])\b",
        r"(?<!\w)g[uü]ne[sş]\s+sistem\w*",
        r"(?<!\w)(?:ka[cç]\s+(?:tane\s+)?gezegen|gezegen\s+say[ıi]s[ıi])\b",
        r"(?<!\w)d[uü]nya(?:'?n[ıi]n|ya)?\s+(?:[sş]ekli|(?:do[gğ]al\s+)?uydusu|en\s+yak[ıi]n\s+y[ıi]ld[ıi]z)\b",
        r"(?<!\w)(?:v[uü]cudumuzdaki|insan\s+v[uü]cudundaki)\s+en\s+b[uü]y[uü]k\s+organ\b",
        r"(?<!\w)(?:g[oö]kku[sş]a[gğ][ıi]nda\s+ka[cç]\s+renk|[uü][cç]genin\s+i[cç]\s+a[cç][ıi]lar[ıi]|karenin\s+ka[cç]\s+kenar[ıi])\b",
        r"(?<!\w)(?:[sş]aka|espri|bilmece)\w*",
        r"(?<!\w)odaklan(?:mak|am[ıi]yorum|maya|abil\w*)(?!\w)",
        r"(?<!\w)(?:motivasyon\w*|stresli\s+(?:hissetti[gğ]imde|oldu[gğ]umda)|uyku\s+d[uü]zen\w*|d[uü]zenli\s+uyumal[ıi]y[ıi]z|neden\s+su\s+i[cç]meliyiz|kitap\s+okuman[ıi]n\s+yararlar[ıi])\b",
    )
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _configure_utf8_stdio() -> None:
    """Safely configure UTF-8 encoding on standard streams when supported."""
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _resolve_index_paths(
    index_dir: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve index directory and artifact paths, respecting overrides."""
    if index_dir is not None:
        target = Path(index_dir)
        return (
            target,
            target / "faiss.index",
            target / "chunks.json",
            target / "embedder.json",
        )

    if INDEX_DIR != _DEFAULT_INDEX_DIR and FAISS_INDEX_PATH == _DEFAULT_INDEX_DIR / "faiss.index":
        target = Path(INDEX_DIR)
        return (
            target,
            target / "faiss.index",
            target / "chunks.json",
            target / "embedder.json",
        )

    return INDEX_DIR, FAISS_INDEX_PATH, CHUNKS_PATH, EMBEDDER_PATH


def load_documents(data_dir: Path) -> list[dict]:
    """Read all .txt files from data_dir. Returns list of {source, text} dicts."""
    data_path = Path(data_dir)
    docs: list[dict] = []
    txt_files = sorted(data_path.glob("*.txt"))
    if not txt_files:
        log.error(f"No .txt files found in '{data_path}'. Add your documents and retry.")
        sys.exit(1)

    for path in txt_files:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"source": path.name, "text": text})
            log.info(f"Loaded: {path.name} ({len(text):,} chars)")

    log.info(f"Loaded {len(docs)} document(s).")
    return docs


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '?', or '!' followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _word_aligned_tail(text: str, overlap: int) -> str:
    """Return a trailing slice of at most `overlap` chars without cutting words in half."""
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text.strip()
    raw_tail = text[-overlap:]
    # If the slice starts mid-word (char right before slice wasn't whitespace), drop the partial word.
    if not text[-overlap - 1].isspace():
        space_pos = raw_tail.find(" ")
        if space_pos != -1:
            return raw_tail[space_pos + 1 :].strip()
    return raw_tail.strip()


def chunk_document(
    doc: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """
    Split a document into overlapping chunks respecting paragraph, sentence, and word boundaries.
    Returns list of {source, chunk_id, text} dicts.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc["text"]) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[dict] = []
    current_chunk = ""
    chunk_id = 0

    def _flush() -> None:
        nonlocal current_chunk, chunk_id
        if current_chunk.strip():
            chunks.append(
                {
                    "source": doc["source"],
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                }
            )
            chunk_id += 1
            current_chunk = ""

    for para in paragraphs:
        if len(para) <= chunk_size:
            # Merge only when at least one side is tiny (< MIN_PARA_MERGE_LEN) so distinct topics stay pure.
            if (
                current_chunk
                and len(current_chunk) + len(para) + 2 <= chunk_size
                and (len(current_chunk) < MIN_PARA_MERGE_LEN or len(para) < MIN_PARA_MERGE_LEN)
            ):
                current_chunk += "\n\n" + para
            else:
                _flush()
                current_chunk = para
            continue

        if current_chunk:
            _flush()

        for sentence in split_into_sentences(para):
            if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
                tail = _word_aligned_tail(current_chunk, overlap)
                _flush()
                current_chunk = f"{tail} {sentence}".strip() if tail else sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence

    _flush()
    return chunks


def build_chunks(
    docs: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """Chunk all documents and return a flat list."""
    all_chunks: list[dict] = []
    for doc in docs:
        doc_chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(doc_chunks)
        log.info(f"  {doc['source']}: {len(doc_chunks)} chunk(s)")
    log.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def load_embedding_model(
    max_features: int = 4096, ngram_range: tuple[int, int] = (3, 4)
) -> LocalTfidfEmbedder:
    """Create a local TF-IDF embedder instance."""
    log.info("Initializing local TF-IDF embedder.")
    return LocalTfidfEmbedder(max_features=max_features, ngram_range=ngram_range)


def build_faiss_index(
    chunks: list[dict], embed_model: LocalTfidfEmbedder
) -> faiss.Index:
    """Fit embed_model (if unfitted) on chunk texts and build a FAISS IP index."""
    if not chunks:
        raise ValueError("Cannot build FAISS index from an empty chunk list.")

    texts = [c["text"] for c in chunks]
    log.info("Generating local TF-IDF embeddings...")
    if getattr(embed_model, "_fitted", False):
        embeddings = embed_model.encode(texts)
    else:
        embeddings = embed_model.fit_transform(texts)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    log.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(
    index: faiss.Index,
    chunks: list[dict],
    embed_model: LocalTfidfEmbedder,
    index_dir: Path | None = None,
) -> None:
    """Persist the FAISS index, chunk metadata, and fitted embedder to disk."""
    target_dir, faiss_path, chunks_path, embedder_path = _resolve_index_paths(index_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    embedder_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(faiss_path))
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    embed_model.save(embedder_path)
    log.info(f"Index saved -> {faiss_path} (embedder -> {embedder_path})")


def load_index(
    index_dir: Path | None = None,
) -> tuple[faiss.Index, list[dict], LocalTfidfEmbedder]:
    """Load a persisted FAISS index, chunk metadata, and fitted embedder."""
    _, faiss_path, chunks_path, embedder_path = _resolve_index_paths(index_dir)
    index = faiss.read_index(str(faiss_path))
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embed_model = LocalTfidfEmbedder.load(embedder_path)
    log.info(f"Index loaded: {index.ntotal} vectors from disk.")
    return index, chunks, embed_model


def index_exists(index_dir: Path | None = None) -> bool:
    """Return True if faiss.index, chunks.json, and embedder.json all exist."""
    _, faiss_path, chunks_path, embedder_path = _resolve_index_paths(index_dir)
    return faiss_path.exists() and chunks_path.exists() and embedder_path.exists()


def retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    embed_model: LocalTfidfEmbedder,
    top_k: int = TOP_K,
    min_score: float = MIN_SCORE,
    min_ratio: float = MIN_RATIO,
) -> list[dict]:
    """Embed the query and return up to top_k chunks passing min_score and relative min_ratio."""
    if not chunks or index.ntotal == 0 or top_k <= 0:
        return []

    query_vec = embed_model.encode([query])
    query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)
    if float(np.linalg.norm(query_vec)) == 0.0:
        return []

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    valid_hits = [
        (float(score), int(idx))
        for score, idx in zip(scores[0], indices[0])
        if idx != -1 and idx < len(chunks)
    ]
    if not valid_hits:
        return []

    best_score = max(score_val for score_val, _ in valid_hits)
    threshold = min_score if min_score <= 0.0 else max(min_score, best_score * min_ratio)

    results: list[dict] = []
    for score_val, idx in valid_hits:
        if score_val < threshold:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = score_val
        results.append(chunk)

    return results


def load_llm(
    model_path: Path | str = MODEL_PATH,
    tokenizer_path: Path | str = TOKENIZER_PATH,
    device: str | None = None,
) -> tuple[MiniLLM, BPETokenizer]:
    """Load the locally trained MiniLLM checkpoint and BPETokenizer."""
    model_file = Path(model_path)
    tokenizer_file = Path(tokenizer_path)

    missing = [p for p in (model_file, tokenizer_file) if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        msg = (
            f"Model veya tokenizer dosyası bulunamadı ({missing_str}). "
            "Lütfen önce 'python train.py' çalıştırın / "
            "Model or tokenizer checkpoint not found. Please run 'python train.py' first."
        )
        log.error(msg)
        raise FileNotFoundError(msg)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info(f"Loading MiniLLM from {model_file} (device={device})")
    tokenizer = BPETokenizer.load(tokenizer_file)
    model, meta = MiniLLM.load_checkpoint(model_file, device=device)
    if meta:
        log.debug(f"Loaded checkpoint metadata: {meta}")
    return model, tokenizer


def _pack_rag_prompt(
    query: str,
    context_chunks: list[dict],
    tokenizer: BPETokenizer,
    block_size: int,
) -> str:
    """Pack retrieved chunks within `block_size - 32` tokens, truncating top-1 if needed."""
    max_prompt_tokens = block_size - 32 if block_size > 32 else max(1, block_size - 8)

    selected_texts: list[str] = []
    non_empty_texts = [c["text"].strip() for c in context_chunks if c.get("text", "").strip()]

    for text in non_empty_texts:
        candidate_texts = selected_texts + [text]
        candidate_context = "\n\n---\n\n".join(candidate_texts)
        candidate_prompt = format_rag_prompt(candidate_context, query)
        if len(tokenizer.encode(candidate_prompt)) <= max_prompt_tokens:
            selected_texts = candidate_texts
        else:
            break

    if selected_texts:
        return format_rag_prompt("\n\n---\n\n".join(selected_texts), query)

    if not non_empty_texts:
        return format_rag_prompt("", query)

    # Top-1 chunk exceeds budget on its own: binary-search its token prefix so it fits cleanly.
    first_text = non_empty_texts[0]
    base_len = len(tokenizer.encode(format_rag_prompt("", query)))
    chunk_ids = tokenizer.encode(first_text)

    lo = 0
    hi = min(len(chunk_ids), max(0, max_prompt_tokens - base_len + 4))
    best_context = ""

    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = tokenizer.decode(chunk_ids[:mid], skip_special_tokens=True).strip()
        prompt_len = len(tokenizer.encode(format_rag_prompt(candidate, query)))
        if prompt_len <= max_prompt_tokens:
            best_context = candidate
            lo = mid + 1
        else:
            hi = mid - 1

    return format_rag_prompt(best_context, query)


def _stream_generate(
    model: MiniLLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_k: int | None = 20,
    stream: bool = True,
    empty_fallback: str = "Bilmiyorum.",
) -> str:
    """Encode prompt, stream tokens from MiniLLM, and return cleaned response text."""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id]
    if len(prompt_ids) > model.config.block_size:
        prompt_ids = prompt_ids[-model.config.block_size :]

    device = next(model.parameters()).device
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids: list[int] = []
    printed_len = 0

    if stream:
        print("\nAsistan: ", end="", flush=True)

    for token_id in model.generate_stream(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_id,
    ):
        generated_ids.append(token_id)
        if stream:
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
            delta = decoded[printed_len:]
            if delta:
                print(delta, end="", flush=True)
                printed_len = len(decoded)

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not answer:
        answer = empty_fallback
        if stream:
            print(answer, end="", flush=True)
    if stream:
        print(flush=True)
    return answer


def generate_answer(
    query: str,
    context_chunks: list[dict],
    llm: tuple[MiniLLM, BPETokenizer],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_k: int | None = 20,
    stream: bool = True,
) -> str:
    """Generate a grounded RAG answer given a query and relevant chunks."""
    if not context_chunks:
        answer = "Malesef aradığınız bilgi elimde yok."
        if stream:
            print(f"\nAsistan: {answer}")
        return answer

    model, tokenizer = llm
    prompt = _pack_rag_prompt(query, context_chunks, tokenizer, model.config.block_size)
    return _stream_generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stream=stream,
        empty_fallback="Bilmiyorum.",
    )


def _normalize_turkish_lower(text: str) -> str:
    """Lowercase text handling Turkish 'İ' -> 'i' cleanly."""
    return text.replace("İ", "i").lower().replace("\u0307", "")


def is_conversational(query: str) -> bool:
    """Detect whether the query is conversational/chitchat rather than a document lookup."""
    q = _normalize_turkish_lower(query).strip()
    if not q:
        return False
    if q in _EXACT_CONVERSATIONAL_QUERIES:
        return True
    q_folded = q.replace("ı", "i")

    for pattern in _CONVERSATIONAL_PATTERNS:
        if pattern.search(q) or pattern.search(q_folded):
            return True

    for kw in CONVERSATIONAL_KEYWORDS:
        kw_norm = _normalize_turkish_lower(kw).strip()
        if not kw_norm:
            continue
        if re.search(rf"(?<!\w){re.escape(kw_norm)}(?!\w)", q):
            return True
        kw_folded = kw_norm.replace("ı", "i")
        if kw_folded != kw_norm and re.search(
            rf"(?<!\w){re.escape(kw_folded)}(?!\w)", q_folded
        ):
            return True
    return False


def generate_chat_response(
    query: str,
    llm: tuple[MiniLLM, BPETokenizer],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_k: int | None = 20,
    stream: bool = True,
) -> str:
    """Generate a conversational response without document retrieval context."""
    model, tokenizer = llm
    prompt = format_chat_prompt(query)
    return _stream_generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stream=stream,
        empty_fallback="...",
    )


def _handle_query(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    embed_model: LocalTfidfEmbedder,
    llm: tuple[MiniLLM, BPETokenizer],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_k: int | None = 20,
) -> str:
    """Route a single query through chat or RAG retrieval and print sources."""
    if is_conversational(query):
        return generate_chat_response(
            query,
            llm,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stream=True,
        )

    relevant_chunks = retrieve(query, index, chunks, embed_model)
    log.debug(f"Retrieved {len(relevant_chunks)} chunks.")

    answer = generate_answer(
        query,
        relevant_chunks,
        llm,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stream=True,
    )

    if relevant_chunks:
        sources = list(dict.fromkeys(c["source"] for c in relevant_chunks))
        print(f"\n  [Kaynaklar: {', '.join(sources)}]\n")
    return answer


def chat_loop(
    index: faiss.Index,
    chunks: list[dict],
    embed_model: LocalTfidfEmbedder,
    llm: tuple[MiniLLM, BPETokenizer],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_k: int | None = 20,
) -> None:
    """Interactive terminal chat loop."""
    print("\n" + "=" * 60)
    print("  Kişisel RAG — Yerel Bilgi Asistanı")
    print("  Sorunuzu yazın. Çıkmak için 'exit' veya 'quit' yazın.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşmek üzere!")
            break

        if not query:
            continue
        if _normalize_turkish_lower(query) in ("exit", "quit", "q"):
            print("Görüşmek üzere!")
            break

        _handle_query(
            query,
            index,
            chunks,
            embed_model,
            llm,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        print("-" * 60)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the local RAG assistant."""
    parser = argparse.ArgumentParser(description="Personal RAG — Local MiniLLM + FAISS")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR,
        help="Path to folder containing .txt files (default: ./data)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the FAISS index even if it already exists",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to MiniLLM checkpoint file (default: ./checkpoints/mini_llm.pt)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=TOKENIZER_PATH,
        help="Path to BPETokenizer JSON file (default: ./checkpoints/tokenizer.json)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query non-interactively and exit",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy decoding)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling cutoff during generation (default: 20)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum number of tokens to generate (default: 128)",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for building/loading the local RAG index and running inference."""
    _configure_utf8_stdio()

    if not torch.cuda.is_available():
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        log.info(f"CPU threads optimized: {num_threads}")

    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    global DATA_DIR
    DATA_DIR = args.data

    if index_exists() and not args.rebuild:
        log.info("Existing index found. Loading from disk (use --rebuild to regenerate).")
        index, chunks, embed_model = load_index()
    else:
        if args.rebuild:
            log.info("--rebuild flag set. Rebuilding index...")
        else:
            log.info("No index found. Building from scratch...")

        embed_model = load_embedding_model()
        docs = load_documents(DATA_DIR)
        chunks = build_chunks(docs)
        index = build_faiss_index(chunks, embed_model)
        save_index(index, chunks, embed_model)

    try:
        llm = load_llm(model_path=args.model, tokenizer_path=args.tokenizer)
    except FileNotFoundError:
        sys.exit(1)

    if args.query is not None:
        query = args.query.strip()
        if not query:
            log.warning("Empty --query provided.")
            return
        _handle_query(
            query,
            index,
            chunks,
            embed_model,
            llm,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        return

    chat_loop(
        index,
        chunks,
        embed_model,
        llm,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
