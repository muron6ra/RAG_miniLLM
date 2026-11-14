"""Comprehensive unit and end-to-end tests for the local zero-dependency RAG pipeline."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import faiss
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rag
from embedder import LocalTfidfEmbedder
from model import MiniLLM, MiniLLMConfig
from tokenizer import (
    AST_TOKEN,
    BOS_TOKEN,
    CTX_TOKEN,
    EOS_TOKEN,
    SPECIAL_TOKENS,
    USR_TOKEN,
    BPETokenizer,
    format_chat_prompt,
    format_rag_prompt,
    format_training_example,
)


def _train_tiny_tokenizer_and_llm(
    block_size: int = 64, steps: int = 25
) -> tuple[MiniLLM, BPETokenizer]:
    """Create and lightly train a tiny MiniLLM + BPETokenizer for fast tests."""
    corpus = [
        "Yapay zeka ve FAISS vektör benzerlik araması yerel çalışır.",
        "Python ile sıfırdan dil modeli eğitilir.",
        "Merhaba! Size nasıl yardımcı olabilirim?",
        "Bu bilgi elimdeki belgelerde yer almıyor.",
    ] * 6

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=320, min_frequency=2)

    config = MiniLLMConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    torch.manual_seed(42)
    model = MiniLLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    p1, t1 = format_training_example(
        "FAISS nedir?",
        "FAISS vektör arama kütüphanesidir.",
        context="FAISS vektör benzerlik araması yapar.",
    )
    p2, t2 = format_training_example(
        "Merhaba!",
        "Merhaba! Nasıl yardımcı olabilirim?",
        context=None,
    )

    seqs = [tokenizer.encode(p1 + t1), tokenizer.encode(p2 + t2)]
    seqs = [s[:block_size] for s in seqs]

    model.train()
    for _ in range(steps):
        for seq in seqs:
            if len(seq) < 2:
                continue
            x = torch.tensor([seq[:-1]], dtype=torch.long)
            y = torch.tensor([seq[1:]], dtype=torch.long)
            _, loss = model(x, y)
            assert loss is not None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model, tokenizer


class TestBPETokenizer(unittest.TestCase):
    """Tests for BPETokenizer training, Turkish roundtrip, special tokens, and serialization."""

    def test_special_tokens_are_atomic(self) -> None:
        texts = [
            f"{CTX_TOKEN} Yapay zeka yerel çalışır. {USR_TOKEN} Yapay zeka nedir? {AST_TOKEN} Bilmiyorum.{EOS_TOKEN}"
        ] * 5
        tok = BPETokenizer()
        tok.train(texts, vocab_size=350, min_frequency=2)

        expected_props = {
            tok.PAD_TOKEN: tok.pad_id,
            tok.BOS_TOKEN: tok.bos_id,
            tok.EOS_TOKEN: tok.eos_id,
            tok.UNK_TOKEN: tok.unk_id,
            tok.CTX_TOKEN: tok.context_id,
            tok.USR_TOKEN: tok.user_id,
            tok.AST_TOKEN: tok.assistant_id,
        }
        self.assertEqual(set(expected_props.keys()), set(SPECIAL_TOKENS))
        for special_str, expected_id in expected_props.items():
            encoded = tok.encode(special_str)
            self.assertEqual(encoded, [expected_id])
            self.assertEqual(tok.decode(encoded, skip_special_tokens=False), special_str)
            self.assertEqual(tok.decode(encoded, skip_special_tokens=True), "")

        mixed = f"{CTX_TOKEN}merhaba{EOS_TOKEN}"
        mixed_ids = tok.encode(mixed)
        self.assertEqual(mixed_ids[0], tok.context_id)
        self.assertEqual(mixed_ids[-1], tok.eos_id)

    def test_train_and_turkish_roundtrip(self) -> None:
        texts = [
            "Türkçe karakterler: çğıöşü ÇĞİÖŞÜ. Yapay zeka modelleri metin işler!\n",
            "Türkçe öğrenen yerel asistan, İngilizce Python kodlarını da anlar.\n",
        ] * 6
        tok = BPETokenizer()
        base_size = tok.vocab_size
        tok.train(texts, vocab_size=base_size + 40, min_frequency=2)

        self.assertGreater(len(tok.merges), 0)
        self.assertGreater(tok.vocab_size, base_size)

        sample = "Çalışkan öğrenci İstanbul'da güneşli öğlen şairliği övdü: çğıöşü ÇĞİÖŞÜ!"
        ids = tok.encode(sample)
        self.assertEqual(tok.decode(ids, skip_special_tokens=True), sample)

        wrapped = tok.encode(sample, add_bos=True, add_eos=True)
        self.assertEqual(wrapped[0], tok.bos_id)
        self.assertEqual(wrapped[-1], tok.eos_id)
        self.assertEqual(tok.decode(wrapped, skip_special_tokens=True), sample)
        self.assertEqual(
            tok.decode(wrapped, skip_special_tokens=False),
            f"{BOS_TOKEN}{sample}{EOS_TOKEN}",
        )

    def test_save_and_load_roundtrip(self) -> None:
        tok = BPETokenizer()
        tok.train(["Merhaba dünya! Yapay zeka ve Türkçe." * 5], vocab_size=300, min_frequency=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            tok.save(path)
            loaded = BPETokenizer.load(path)

        self.assertEqual(loaded.vocab, tok.vocab)
        self.assertEqual(loaded.merges, tok.merges)
        self.assertEqual(loaded.vocab_size, tok.vocab_size)

        probe = f"{USR_TOKEN} Güneşli bir İstanbul günü! {AST_TOKEN}"
        self.assertEqual(loaded.encode(probe), tok.encode(probe))
        self.assertEqual(
            loaded.decode(loaded.encode(probe), skip_special_tokens=False),
            probe,
        )

    def test_prompt_formatters(self) -> None:
        chat_p = format_chat_prompt("  Nasılsın?  ")
        self.assertEqual(chat_p, f"{USR_TOKEN} Nasılsın? {AST_TOKEN} ")

        rag_p = format_rag_prompt("  Bağlam metni. ", " Soru nedir? ")
        self.assertEqual(rag_p, f"{CTX_TOKEN} Bağlam metni. {USR_TOKEN} Soru nedir? {AST_TOKEN} ")

        p_rag, t_rag = format_training_example("Soru", "Cevap", context="Bağlam")
        self.assertEqual(p_rag, format_rag_prompt("Bağlam", "Soru"))
        self.assertEqual(t_rag, f"Cevap{EOS_TOKEN}")

        p_chat, t_chat = format_training_example("Selam", "Merhaba!", context="   ")
        self.assertEqual(p_chat, format_chat_prompt("Selam"))
        self.assertEqual(t_chat, f"Merhaba!{EOS_TOKEN}")


class TestLocalTfidfEmbedder(unittest.TestCase):
    """Tests for LocalTfidfEmbedder normalization, ranking, zero-norm safety, and persistence."""

    def test_fit_encode_l2_normalized_float32(self) -> None:
        embedder = LocalTfidfEmbedder(max_features=256, ngram_range=(3, 4))
        with self.assertRaises(ValueError):
            embedder.encode(["test"])
        with self.assertRaises(ValueError):
            embedder.fit([])

        docs = [
            "Yapay zeka ve derin öğrenme modelleri.",
            "FAISS vektör benzerlik araması yapar.",
            "Python programlama dili sade sözdizimine sahiptir.",
        ]
        mat = embedder.fit_transform(docs)
        self.assertEqual(mat.dtype, np.float32)
        self.assertEqual(mat.shape, (3, embedder.dim))
        self.assertLessEqual(embedder.dim, 256)

        norms = np.linalg.norm(mat, axis=1)
        for norm in norms:
            self.assertAlmostEqual(float(norm), 1.0, places=5)

        empty_batch = embedder.encode([])
        self.assertEqual(empty_batch.shape, (0, embedder.dim))
        self.assertEqual(empty_batch.dtype, np.float32)

    def test_turkish_and_english_similarity_ranking(self) -> None:
        docs = [
            "İstanbul Boğazı tarihi yarımada ve kültür şehridir.",
            "FAISS is a library for efficient vector similarity search.",
            "Yapay zeka doğal dil işleme ve makine öğrenmesi alanıdır.",
            "Python is a general-purpose programming language with clean syntax.",
        ]
        embedder = LocalTfidfEmbedder(max_features=1024)
        doc_vecs = embedder.fit_transform(docs)

        tr_upper_query = embedder.encode(["İSTANBUL tarihi"])
        scores_tr = (doc_vecs @ tr_upper_query.T).squeeze(1)
        self.assertEqual(int(np.argmax(scores_tr)), 0)

        en_query = embedder.encode(["efficient vector similarity search library"])
        scores_en = (doc_vecs @ en_query.T).squeeze(1)
        self.assertEqual(int(np.argmax(scores_en)), 1)

        nlp_query = embedder.encode(["doğal dil işleme makine öğrenmesi"])
        scores_nlp = (doc_vecs @ nlp_query.T).squeeze(1)
        self.assertEqual(int(np.argmax(scores_nlp)), 2)

    def test_zero_norm_safety(self) -> None:
        embedder = LocalTfidfEmbedder(max_features=128)
        embedder.fit(["alpha beta gamma", "delta epsilon zeta"])

        vecs = embedder.encode(["", "   ", "!!!???", "99999999"])
        self.assertEqual(vecs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(vecs)))
        np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 0.0, atol=1e-7)

    def test_save_and_load_roundtrip(self) -> None:
        unfitted = LocalTfidfEmbedder()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embedder.json"
            with self.assertRaises(ValueError):
                unfitted.save(path)

            embedder = LocalTfidfEmbedder(max_features=512, ngram_range=(3, 4))
            embedder.fit(["Türkiye'de yapay zeka", "Python and FAISS retrieval"])
            embedder.save(path)

            loaded = LocalTfidfEmbedder.load(path)

        self.assertEqual(loaded.dim, embedder.dim)
        self.assertEqual(loaded.vocab, embedder.vocab)
        probe = ["İstanbul yapay zeka", "FAISS retrieval"]
        np.testing.assert_allclose(loaded.encode(probe), embedder.encode(probe), atol=1e-6)


class TestMiniLLM(unittest.TestCase):
    """Tests for MiniLLM config, forward/loss masking, generation, penalty, and checkpoints."""

    def test_config_validation(self) -> None:
        for kwargs in (
            {"vocab_size": 0},
            {"vocab_size": 10, "block_size": 0},
            {"vocab_size": 10, "n_layer": 0},
            {"vocab_size": 10, "n_head": 0},
            {"vocab_size": 10, "n_head": 4, "n_embd": 30},
        ):
            with self.assertRaises(ValueError):
                MiniLLMConfig(**kwargs)

    def test_forward_and_masked_loss(self) -> None:
        config = MiniLLMConfig(
            vocab_size=24, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0
        )
        torch.manual_seed(0)
        model = MiniLLM(config)
        model.eval()

        idx = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.long)
        logits, loss_none = model(idx)
        self.assertIsNone(loss_none)
        self.assertEqual(tuple(logits.shape), (2, 5, 24))

        targets = torch.tensor(
            [[2, 3, -100, -100, -100], [7, -100, -100, -100, -100]],
            dtype=torch.long,
        )
        logits_2, loss = model(idx, targets)
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))

        # Verify -100 targets are excluded identically to explicit cross_entropy slicing.
        valid_logits = torch.cat([logits_2[0, :2], logits_2[1, :1]], dim=0)
        valid_targets = torch.tensor([2, 3, 7], dtype=torch.long)
        expected_loss = F.cross_entropy(valid_logits, valid_targets)
        self.assertAlmostEqual(float(loss.item()), float(expected_loss.item()), places=5)

        too_long = torch.zeros((1, config.block_size + 1), dtype=torch.long)
        with self.assertRaises(ValueError):
            model(too_long)

    def test_generate_stream_and_eos_stopping(self) -> None:
        config = MiniLLMConfig(
            vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0
        )
        torch.manual_seed(7)
        model = MiniLLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)

        eos_id = 2
        seq = [4, 5, 6, eos_id]
        x = torch.tensor([seq[:-1]], dtype=torch.long)
        y = torch.tensor([seq[1:]], dtype=torch.long)

        model.train()
        for _ in range(40):
            _, loss = model(x, y)
            assert loss is not None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()

        prompt = torch.tensor([[4, 5]], dtype=torch.long)
        streamed = list(
            model.generate_stream(
                prompt,
                max_new_tokens=10,
                temperature=0.0,
                eos_token_id=eos_id,
            )
        )
        self.assertEqual(streamed, [6])
        self.assertNotIn(eos_id, streamed)

        full = model.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.0,
            eos_token_id=eos_id,
        )
        self.assertEqual(full.tolist(), [[4, 5, 6]])

    def test_repetition_penalty_only_affects_generated_tokens(self) -> None:
        config = MiniLLMConfig(
            vocab_size=12, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0
        )
        model = MiniLLM(config)
        model.eval()

        tok_in_prompt = 5
        tok_alt = 6

        def fake_forward(idx: torch.Tensor, targets: torch.Tensor | None = None):
            bsz, seq_len = idx.size()
            logits = torch.full((bsz, seq_len, config.vocab_size), -5.0)
            logits[:, :, tok_in_prompt] = 10.0
            logits[:, :, tok_alt] = 6.0
            return logits, None

        prompt = torch.tensor([[tok_in_prompt, 3, 4]], dtype=torch.long)
        with mock.patch.object(model, "forward", side_effect=fake_forward):
            # Step 1: tok_in_prompt is in prompt only, NOT generated yet -> unpenalized -> 5.
            # Step 2: 5 is now generated -> 10.0 / 5.0 = 2.0 < 6.0 -> selects tok_alt (6).
            out_penalized = list(
                model.generate_stream(
                    prompt,
                    max_new_tokens=2,
                    temperature=0.0,
                    repetition_penalty=5.0,
                )
            )
            self.assertEqual(out_penalized, [tok_in_prompt, tok_alt])

            out_unpenalized = list(
                model.generate_stream(
                    prompt,
                    max_new_tokens=2,
                    temperature=0.0,
                    repetition_penalty=1.0,
                )
            )
            self.assertEqual(out_unpenalized, [tok_in_prompt, tok_in_prompt])

    def test_save_and_load_checkpoint(self) -> None:
        config = MiniLLMConfig(
            vocab_size=30, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0
        )
        torch.manual_seed(11)
        model = MiniLLM(config)
        model.eval()

        probe = torch.tensor([[1, 4, 7, 2]], dtype=torch.long)
        with torch.no_grad():
            orig_logits, _ = model(probe)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "mini_llm.pt"
            model.save_checkpoint(ckpt, meta={"epoch": 3, "val_loss": 0.42})
            loaded_model, meta = MiniLLM.load_checkpoint(ckpt, device="cpu")

        self.assertFalse(loaded_model.training)
        self.assertEqual(meta, {"epoch": 3, "val_loss": 0.42})
        with torch.no_grad():
            loaded_logits, _ = loaded_model(probe)
        self.assertTrue(torch.allclose(orig_logits, loaded_logits))


class TestRagPipeline(unittest.TestCase):
    """End-to-end tests for rag.py pipeline functions in isolated temporary directories."""

    def test_load_documents_and_chunking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "tr.txt").write_text(
                "Yapay zeka çok hızlı gelişiyor. FAISS vektör arama yapar! Peki Python neden seçildi?",
                encoding="utf-8",
            )
            (data_dir / "en.txt").write_text(
                "Python is widely used in machine learning. Transformers use self-attention.",
                encoding="utf-8",
            )

            docs = rag.load_documents(data_dir)
            self.assertEqual(len(docs), 2)
            self.assertEqual([d["source"] for d in docs], ["en.txt", "tr.txt"])

            sents = rag.split_into_sentences(docs[1]["text"])
            self.assertEqual(len(sents), 3)

            chunks = rag.build_chunks(docs, chunk_size=60, overlap=10)
            self.assertGreater(len(chunks), 2)
            for chunk in chunks:
                self.assertIn("source", chunk)
                self.assertIn("chunk_id", chunk)
                self.assertIn("text", chunk)

            zero_overlap_chunks = rag.chunk_document(docs[1], chunk_size=45, overlap=0)
            self.assertGreater(len(zero_overlap_chunks), 1)
            self.assertNotIn(zero_overlap_chunks[0]["text"], zero_overlap_chunks[1]["text"])

            # Tiny paragraphs < 120 chars merge; distinct paragraphs >= 120 chars stay single-topic.
            para_tiny = "Kısa başlık."
            para_a = "Yapay zeka sistemleri insan zekasını taklit eder ve verilerden örüntü çıkararak doğal dili anlamayı başarır." + " " + ("ek bilgi " * 4)
            para_b = "FAISS kütüphanesi yoğun vektör kümelerinde hızlı kosinüs benzerliği ve iç çarpım araması gerçekleştirmektedir." + " " + ("ek bilgi " * 4)
            multi_para_doc = {
                "source": "multi.txt",
                "text": f"{para_tiny}\n\n{para_a.strip()}\n\n{para_b.strip()}",
            }
            para_chunks = rag.chunk_document(multi_para_doc, chunk_size=750, overlap=80)
            self.assertEqual(len(para_chunks), 2)
            self.assertIn(para_tiny, para_chunks[0]["text"])
            self.assertIn(para_a.strip(), para_chunks[0]["text"])
            self.assertEqual(para_chunks[1]["text"], para_b.strip())

    def test_faiss_build_save_load_and_retrieve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_dir = tmp_path / "data"
            index_dir = tmp_path / "index"
            data_dir.mkdir()

            (data_dir / "bilgi_tr.txt").write_text(
                "FAISS vektör benzerlik araması yapar. Kosinüs benzerliği ile en yakın parçalar bulunur.",
                encoding="utf-8",
            )
            (data_dir / "python_en.txt").write_text(
                "Python is a readable general-purpose programming language with rich libraries.",
                encoding="utf-8",
            )
            (data_dir / "yapay_zeka.txt").write_text(
                "Transformer mimarisi öz-dikkat mekanizması ile metin üretir.",
                encoding="utf-8",
            )

            docs = rag.load_documents(data_dir)
            chunks = rag.build_chunks(docs)
            embed_model = LocalTfidfEmbedder(max_features=1024)
            index = rag.build_faiss_index(chunks, embed_model)

            self.assertIsInstance(index, faiss.IndexFlatIP)
            self.assertEqual(index.ntotal, len(chunks))
            self.assertFalse(rag.index_exists(index_dir=index_dir))

            rag.save_index(index, chunks, embed_model, index_dir=index_dir)
            self.assertTrue(rag.index_exists(index_dir=index_dir))
            self.assertTrue((index_dir / "faiss.index").exists())
            self.assertTrue((index_dir / "chunks.json").exists())
            self.assertTrue((index_dir / "embedder.json").exists())

            loaded_index, loaded_chunks, loaded_embedder = rag.load_index(index_dir=index_dir)
            self.assertEqual(loaded_index.ntotal, len(chunks))
            self.assertEqual(loaded_chunks, chunks)

            # Query with top_k > ntotal verifies idx == -1 filtering and min_score filtering
            # (other chunks have low positive noise scores < 0.05 and must be dropped).
            all_hits_tr = rag.retrieve(
                "FAISS vektör benzerlik araması",
                loaded_index,
                loaded_chunks,
                loaded_embedder,
                top_k=5,
                min_score=0.0,
            )
            self.assertEqual(len(all_hits_tr), 3)
            self.assertLess(all_hits_tr[-1]["score"], 0.05)

            hits_tr = rag.retrieve(
                "FAISS vektör benzerlik araması",
                loaded_index,
                loaded_chunks,
                loaded_embedder,
                top_k=5,
                min_score=0.05,
            )
            self.assertEqual(len(hits_tr), 1)
            self.assertEqual(hits_tr[0]["source"], "bilgi_tr.txt")
            self.assertGreaterEqual(hits_tr[0]["score"], 0.05)

            hits_en = rag.retrieve(
                "Python programming language",
                loaded_index,
                loaded_chunks,
                loaded_embedder,
                top_k=2,
                min_score=0.05,
            )
            self.assertGreaterEqual(len(hits_en), 1)
            self.assertEqual(hits_en[0]["source"], "python_en.txt")

            unrelated = rag.retrieve(
                "çikolata",
                loaded_index,
                loaded_chunks,
                loaded_embedder,
                top_k=4,
                min_score=0.05,
            )
            self.assertEqual(unrelated, [])

            zero_norm = rag.retrieve(
                "99999999",
                loaded_index,
                loaded_chunks,
                loaded_embedder,
                top_k=4,
                min_score=0.05,
            )
            self.assertEqual(zero_norm, [])

            # Relative min_ratio drops weak secondary hits while preserving close co-hits.
            ratio_chunks = [
                {
                    "source": "strong.txt",
                    "chunk_id": 0,
                    "text": "FAISS yoğun vektör benzerlik araması kütüphanesidir ve hızlı arama yapar.",
                },
                {
                    "source": "weak.txt",
                    "chunk_id": 0,
                    "text": "Python genel amaçlı okunabilir bir programlama dilidir ve hızlı geliştirme sağlar.",
                },
            ]
            ratio_emb = LocalTfidfEmbedder(max_features=512)
            ratio_idx = rag.build_faiss_index(ratio_chunks, ratio_emb)
            unfiltered = rag.retrieve(
                "FAISS vektör benzerlik araması hızlı",
                ratio_idx,
                ratio_chunks,
                ratio_emb,
                top_k=2,
                min_score=0.01,
                min_ratio=0.0,
            )
            self.assertEqual(len(unfiltered), 2)
            self.assertLess(unfiltered[1]["score"], unfiltered[0]["score"] * 0.55)
            filtered = rag.retrieve(
                "FAISS vektör benzerlik araması hızlı",
                ratio_idx,
                ratio_chunks,
                ratio_emb,
                top_k=2,
                min_score=0.01,
                min_ratio=0.55,
            )
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0]["source"], "strong.txt")

    def test_is_conversational_turkish_and_english(self) -> None:
        for q in (
            "Merhaba!",
            "İYİ GÜNLER",
            "GÜNAYDIN",
            "Nasılsın?",
            "TEŞEKKÜRLER",
            "Görüşürüz",
            "Sen kimsin?",
            "Hello there",
        ):
            self.assertTrue(rag.is_conversational(q), msg=f"Expected conversational: {q!r}")

        for q in (
            "What is machine learning?",
            "Yapay zeka nedir?",
            "FAISS nasıl çalışır?",
            "Python hakkında bilgi ver",
            "They use transformers",
        ):
            self.assertFalse(rag.is_conversational(q), msg=f"Expected RAG query: {q!r}")

    def test_is_conversational_everyday_paraphrases_and_rag_isolation(self) -> None:
        everyday_queries = (
            "merhaba",
            "selamlar nasılsın",
            "sen kimsin",
            "ismin ne senin",
            "ISMIN NE SENIN?",
            "neler yapabiliyorsun",
            "Neler yapabilirsiniz?",
            "saat kaç acaba",
            "hava durumu nasıl bugün",
            "bugün tarih nedir acaba",
            "türkiye başkenti neresi",
            "Türkiye'nin başkenti neresidir?",
            "haftada kaç gün var",
            "bir senede kaç ay vardır",
            "su kaç derecede kaynar acaba",
            "su kaç derecede donar",
            "güneş sisteminde kaç tane gezegen var",
            "verilerim güvende mi",
            "bilgilerim güvende mi?",
            "bana şaka yapar mısın",
            "Bana saka yapar misin?",
            "bana bilmece sorar mısın",
            "odaklanmak için ne yapmalıyım",
            "çok teşekkür ederim sana",
            "Cok tesekkur ederim",
            "şimdilik hoşça kal kendine dikkat et",
            "sa",
            "???",
        )
        for q in everyday_queries:
            self.assertTrue(rag.is_conversational(q), msg=f"Expected conversational: {q!r}")

        rag_queries = (
            "Yapay zeka nedir?",
            "RAG nedir?",
            "RAG nedir ve neden kullanılır?",
            "FAISS hangi dille yazılmıştır?",
            "Bu projede MiniLM veya TinyLlama kullanılıyor mu?",
            "Python neden bu projenin ana programlama dili seçilmiştir?",
            "Bu sistemde veri gizliliği nasıl korunur?",
            "Embedding işlemi metinleri neye dönüştürür?",
            "Transformer mimarisi ne zaman tanıtıldı?",
            "Sentence Transformers nedir?",
            "What is Artificial Intelligence?",
            "What is machine learning?",
            "Deep learning architectures",
            "Makine öğrenmesi neye odaklanır?",
            "Python neler yapar?",
            "FAISS ne işe yarar?",
            "İtalya'nın başkenti neresidir?",
            "Mars gezegeninin yüzey sıcaklığı kaç derecedir?",
            "Menemen tarifi nedir?",
            "Bitcoin fiyatı bugün kaç dolar?",
            "RAG sisteminde hangi ağ portu kullanılır?",
            "saatlerin çalışma prensibi",
            "hesap makinesi nedir?",
        )
        for q in rag_queries:
            self.assertFalse(rag.is_conversational(q), msg=f"Expected RAG query: {q!r}")

    def test_load_llm_missing_files_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                rag.load_llm(
                    model_path=tmp / "missing_model.pt",
                    tokenizer_path=tmp / "missing_tok.json",
                )

    def test_end_to_end_rag_and_chat_generation(self) -> None:
        model, tokenizer = _train_tiny_tokenizer_and_llm(block_size=64, steps=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            model_path = tmp / "mini_llm.pt"
            tok_path = tmp / "tokenizer.json"
            model.save_checkpoint(model_path)
            tokenizer.save(tok_path)

            llm = rag.load_llm(model_path=model_path, tokenizer_path=tok_path, device="cpu")

            no_ctx_ans = rag.generate_answer("Bilinmeyen soru?", [], llm, stream=False)
            self.assertEqual(no_ctx_ans, "Malesef aradığınız bilgi elimde yok.")

            chunks = [
                {"source": "doc.txt", "chunk_id": 0, "text": "FAISS vektör benzerlik araması yapar."}
            ]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rag_ans = rag.generate_answer(
                    "FAISS nedir?",
                    chunks,
                    llm,
                    max_new_tokens=16,
                    temperature=0.0,
                    stream=True,
                )
            self.assertIsInstance(rag_ans, str)
            self.assertTrue(len(rag_ans) > 0)
            self.assertIn("Asistan:", buf.getvalue())

            # Context packing must enforce <= block_size - 32 even with giant top-1 chunks.
            huge_chunks = [
                {
                    "source": "huge.txt",
                    "chunk_id": 0,
                    "text": ("Yapay zeka ve FAISS vektör arama sistemi yerel olarak çalışır. " * 25),
                },
                {
                    "source": "huge.txt",
                    "chunk_id": 1,
                    "text": "İkinci parça.",
                },
            ]
            packed_prompt = rag._pack_rag_prompt(
                "FAISS nedir?", huge_chunks, tokenizer, model.config.block_size
            )
            self.assertLessEqual(
                len(tokenizer.encode(packed_prompt)), model.config.block_size - 32
            )
            truncated_ans = rag.generate_answer(
                "FAISS nedir?",
                huge_chunks,
                llm,
                max_new_tokens=8,
                temperature=0.0,
                stream=False,
            )
            self.assertIsInstance(truncated_ans, str)
            self.assertTrue(len(truncated_ans) > 0)

            chat_ans = rag.generate_chat_response(
                "Merhaba!", llm, max_new_tokens=16, temperature=0.0, stream=False
            )
            self.assertIsInstance(chat_ans, str)
            self.assertTrue(len(chat_ans) > 0)

    def test_main_single_query_cli_isolated(self) -> None:
        model, tokenizer = _train_tiny_tokenizer_and_llm(block_size=64, steps=15)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_dir = tmp / "data"
            index_dir = tmp / "index"
            ckpt_dir = tmp / "checkpoints"
            data_dir.mkdir()
            ckpt_dir.mkdir()

            (data_dir / "doc.txt").write_text(
                "FAISS vektör benzerlik araması yapar.", encoding="utf-8"
            )
            model_path = ckpt_dir / "mini_llm.pt"
            tok_path = ckpt_dir / "tokenizer.json"
            model.save_checkpoint(model_path)
            tokenizer.save(tok_path)

            argv_rag = [
                "rag.py",
                "--data", str(data_dir),
                "--rebuild",
                "--model", str(model_path),
                "--tokenizer", str(tok_path),
                "--query", "FAISS nedir?",
                "--max-new-tokens", "8",
            ]
            out_buf = io.StringIO()
            with (
                mock.patch.object(rag, "INDEX_DIR", index_dir),
                mock.patch.object(rag, "DATA_DIR", data_dir),
                mock.patch.object(sys, "argv", argv_rag),
                contextlib.redirect_stdout(out_buf),
            ):
                rag.main()

            self.assertTrue((index_dir / "faiss.index").exists())
            self.assertTrue((index_dir / "chunks.json").exists())
            self.assertTrue((index_dir / "embedder.json").exists())
            self.assertIn("Asistan:", out_buf.getvalue())
            self.assertIn("[Kaynaklar: doc.txt]", out_buf.getvalue())

            # Second run without --rebuild exercises load_index() + conversational --query.
            argv_chat = [
                "rag.py",
                "--data", str(data_dir),
                "--model", str(model_path),
                "--tokenizer", str(tok_path),
                "--query", "Merhaba!",
                "--max-new-tokens", "8",
            ]
            chat_buf = io.StringIO()
            with (
                mock.patch.object(rag, "INDEX_DIR", index_dir),
                mock.patch.object(rag, "DATA_DIR", data_dir),
                mock.patch.object(sys, "argv", argv_chat),
                contextlib.redirect_stdout(chat_buf),
            ):
                rag.main()

            self.assertIn("Asistan:", chat_buf.getvalue())
            self.assertNotIn("[Kaynaklar:", chat_buf.getvalue())

    def test_chat_loop_interactive_session_flow(self) -> None:
        model, tokenizer = _train_tiny_tokenizer_and_llm(block_size=64, steps=15)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_dir = tmp / "data"
            data_dir.mkdir()
            (data_dir / "doc.txt").write_text(
                "FAISS vektör benzerlik araması yapar.", encoding="utf-8"
            )

            docs = rag.load_documents(data_dir)
            chunks = rag.build_chunks(docs)
            embed_model = LocalTfidfEmbedder(max_features=256)
            index = rag.build_faiss_index(chunks, embed_model)
            llm = (model, tokenizer)

            scripted_inputs = io.StringIO(
                "\nmerhaba\nFAISS nedir?\nçikolata tarifi nedir?\nexit\n"
            )
            out_buf = io.StringIO()
            with (
                mock.patch.object(sys, "stdin", scripted_inputs),
                contextlib.redirect_stdout(out_buf),
            ):
                rag.chat_loop(
                    index,
                    chunks,
                    embed_model,
                    llm,
                    max_new_tokens=8,
                    temperature=0.0,
                )

            output = out_buf.getvalue()
            self.assertIn("Kişisel RAG — Yerel Bilgi Asistanı", output)
            self.assertGreaterEqual(output.count("Sen: "), 5)
            self.assertGreaterEqual(output.count("Asistan:"), 3)
            self.assertIn("[Kaynaklar: doc.txt]", output)
            self.assertIn("Malesef aradığınız bilgi elimde yok.", output)
            self.assertIn("Görüşmek üzere!", output)

            # Empty stdin (EOF) and Turkish uppercase EXİT should both exit cleanly.
            eof_buf = io.StringIO()
            with (
                mock.patch.object(sys, "stdin", io.StringIO("")),
                contextlib.redirect_stdout(eof_buf),
            ):
                rag.chat_loop(index, chunks, embed_model, llm, max_new_tokens=4)
            self.assertIn("Görüşmek üzere!", eof_buf.getvalue())

            upper_exit_buf = io.StringIO()
            with (
                mock.patch.object(sys, "stdin", io.StringIO("EXİT\n")),
                contextlib.redirect_stdout(upper_exit_buf),
            ):
                rag.chat_loop(index, chunks, embed_model, llm, max_new_tokens=4)
            self.assertIn("Görüşmek üzere!", upper_exit_buf.getvalue())

    def test_parse_args_and_no_external_dependencies(self) -> None:
        defaults = rag.parse_args([])
        self.assertEqual(defaults.data, rag.DATA_DIR)
        self.assertFalse(defaults.rebuild)
        self.assertFalse(defaults.debug)
        self.assertEqual(defaults.model, rag.MODEL_PATH)
        self.assertEqual(defaults.tokenizer, rag.TOKENIZER_PATH)
        self.assertIsNone(defaults.query)
        self.assertEqual(defaults.temperature, 0.0)
        self.assertEqual(defaults.top_k, 20)
        self.assertEqual(defaults.max_new_tokens, 128)

        custom = rag.parse_args(
            [
                "--data", "custom_data",
                "--rebuild",
                "--debug",
                "--model", "m.pt",
                "--tokenizer", "t.json",
                "--query", "Yapay zeka nedir?",
                "--temperature", "0.5",
                "--top-k", "10",
                "--max-new-tokens", "64",
            ]
        )
        self.assertEqual(custom.data, Path("custom_data"))
        self.assertTrue(custom.rebuild)
        self.assertTrue(custom.debug)
        self.assertEqual(custom.model, Path("m.pt"))
        self.assertEqual(custom.tokenizer, Path("t.json"))
        self.assertEqual(custom.query, "Yapay zeka nedir?")
        self.assertEqual(custom.temperature, 0.5)
        self.assertEqual(custom.top_k, 10)
        self.assertEqual(custom.max_new_tokens, 64)

        rag_src = (PROJECT_ROOT / "rag.py").read_text(encoding="utf-8")
        self.assertNotIn("sentence_transformers", rag_src)
        self.assertNotIn("transformers", rag_src)
        self.assertNotIn("TinyLlama", rag_src)

        req_src = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertNotIn("sentence-transformers", req_src)
        self.assertNotIn("transformers", req_src)
        self.assertNotIn("accelerate", req_src)


if __name__ == "__main__":
    unittest.main()
