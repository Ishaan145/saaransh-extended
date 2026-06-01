"""
saaransh-unified.nlp.summarizer
===============================
Hierarchical BART-Large-CNN summariser described in paper §4.1.

Two-stage:
1. Comments are bucketed into groups of `CLUSTER_SIZE` (default 50) and
   each bucket is summarised to ~200 tokens.
2. The intermediate summaries are concatenated and re-summarised to
   ~150 tokens.

Backends mirror the sentiment module:
* BartBackend -- real `facebook/bart-large-cnn` when weights present.
* ExtractiveBackend -- TF-IDF top-k sentence selector fallback.
"""

from __future__ import annotations
import os
import re
import math
from collections import Counter
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
CLUSTER_SIZE = 50


def _split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return sents


class ExtractiveBackend:
    """TF-IDF extractive fallback. Selects the top-k sentences with
    highest summed TF-IDF score against the bucket's vocabulary."""

    def __init__(self, top_k_intermediate: int = 4, top_k_final: int = 5):
        self.top_k_intermediate = top_k_intermediate
        self.top_k_final = top_k_final

    def _tfidf_topk(self, sentences: List[str], k: int) -> List[str]:
        if len(sentences) <= k:
            return sentences
        toks = [re.findall(r"[a-zA-Z]+", s.lower()) for s in sentences]
        df = Counter()
        for s in toks:
            for w in set(s):
                df[w] += 1
        n = len(sentences)
        scores = []
        for s, t in zip(sentences, toks):
            tf = Counter(t)
            sc = 0.0
            for w, c in tf.items():
                sc += c * math.log((1 + n) / (1 + df[w]))
            scores.append(sc / max(1, len(t)))
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
        order.sort()
        return [sentences[i] for i in order]

    def summarise(self, text: str, k: int) -> str:
        return " ".join(self._tfidf_topk(_split_sentences(text), k))

    def hierarchical(self, comments: List[str]) -> dict:
        buckets = [comments[i : i + CLUSTER_SIZE] for i in range(0, len(comments), CLUSTER_SIZE)]
        intermediates = []
        for b in buckets:
            joined = " ".join(b)
            intermediates.append(self.summarise(joined, self.top_k_intermediate))
        final = self.summarise(" ".join(intermediates), self.top_k_final)
        return {
            "backend": "extractive",
            "n_buckets": len(buckets),
            "intermediates": intermediates,
            "final": final,
        }


class BartBackend:
    """Real BART-Large-CNN. Lazy import."""

    def __init__(self, model_dir: str):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @staticmethod
    def is_available(model_dir: str) -> bool:
        if not model_dir or not os.path.isdir(model_dir):
            return False
        files = os.listdir(model_dir)
        return any(f.endswith(".bin") or f.endswith(".safetensors") for f in files)

    def _gen(self, text: str, max_tok: int) -> str:
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        out = self.model.generate(**enc, max_length=max_tok, num_beams=4, length_penalty=2.0)
        return self.tok.decode(out[0], skip_special_tokens=True)

    def hierarchical(self, comments: List[str]) -> dict:
        buckets = [comments[i : i + CLUSTER_SIZE] for i in range(0, len(comments), CLUSTER_SIZE)]
        intermediates = [self._gen(" ".join(b), 200) for b in buckets]
        final = self._gen(" ".join(intermediates), 150)
        return {
            "backend": "bart",
            "n_buckets": len(buckets),
            "intermediates": intermediates,
            "final": final,
        }


def load_summarizer(model_dir: str | None = None):
    if model_dir and BartBackend.is_available(model_dir):
        return BartBackend(model_dir)
    return ExtractiveBackend()
