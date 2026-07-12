"""
web.py — Web fetch + RAG HD (retrieval-augmented generation in HD space).

PROBLEM
-------
AETHER's KB is small (only what it's taught). For real-world questions
like "What is the population of Tokyo?" the agent needs to fetch external
knowledge. Standard LLMs use vector databases (Pinecone, FAISS) with
embedding models (OpenAI, BGE).

AETHER'S SOLUTION
-----------------
HD-native RAG:

  1. Web fetch: query a search endpoint (or local cache) for the question.
  2. HD encoding: encode each retrieved paragraph as an HD vector
     (using the char n-gram + token bundling encoder).
  3. HD indexing: store paragraphs in the SDM with their HD vector as address.
  4. HD retrieval: at query time, encode the question as HD, read from SDM,
     get the most similar paragraph.
  5. Augmented response: feed the retrieved paragraph to the generator.

No embedding model. No vector database. No GPU. Just HD vectors + SDM.

GRACEFUL OFFLINE FALLBACK
-------------------------
If no internet is available (or the fetch fails), the agent falls back to
a curated "wiki" cache — a small built-in knowledge base of common topics
that demonstrates the RAG pipeline without external dependencies.
"""

from __future__ import annotations
import json
import os
import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from .hd import HDVector, DIM, bundle, _sign
from .memory import AssociativeMemory, SparseDistributedMemory
from .encoder import TextEncoder, tokenize


# ---------------------------------------------------------------------------
# Curated offline wiki (used as fallback when no internet)
# ---------------------------------------------------------------------------

# Each entry: (topic, paragraph). The topic is used as a query hint;
# the paragraph is what gets encoded and stored.
OFFLINE_WIKI: List[Tuple[str, str]] = [
    ("tokyo",
     "Tokyo is the capital city of Japan. It has a population of approximately 14 million people in the city proper, and 37 million in the greater metropolitan area, making it the most populous metropolitan area in the world. Tokyo was formerly known as Edo."),
    ("paris",
     "Paris is the capital city of France. It has a population of approximately 2.2 million people in the city proper, and 12 million in the metropolitan area. Paris is known as the City of Light and is famous for the Eiffel Tower, the Louvre museum, and Notre-Dame cathedral."),
    ("japan",
     "Japan is an island country in East Asia. It has a population of approximately 125 million people. Japan is known for its technology, anime, sushi, Mount Fuji, and traditional culture. The currency is the Japanese yen."),
    ("france",
     "France is a country in Western Europe. It has a population of approximately 67 million people. France is known for its wine, cheese, art, fashion, and the Eiffel Tower. The capital is Paris and the currency is the euro."),
    ("canada",
     "Canada is a country in North America. It has a population of approximately 38 million people. Canada is known for its maple syrup, hockey, vast wilderness, and bilingualism (English and French). The capital is Ottawa."),
    ("python",
     "Python is a high-level programming language created by Guido van Rossum in 1991. It is known for its clean syntax, dynamic typing, and large standard library. Python is widely used in data science, machine learning, web development, and automation."),
    ("aether",
     "AETHER (Adaptive Emergent Thinking Hyperdimensional Engine for Reasoning) is a non-transformer cognitive architecture that uses hyperdimensional computing, sparse distributed memory, and a continuous cognitive loop. It learns instantly without backpropagation, runs on CPU, and uses vector symbolic architecture for representation."),
    ("hyperdimensional",
     "Hyperdimensional computing (HDC) is a paradigm that uses very large vectors (typically 1000-10000 dimensions) of bipolar or real values to represent concepts. Operations like binding (XOR), bundling (majority vote), and permutation form an algebra that can encode structured knowledge. HDC is biologically inspired and noise-robust."),
    ("kanerva",
     "Pentti Kanerva introduced Sparse Distributed Memory in 1988. It stores data at sparse hard locations in a high-dimensional space, with reads and writes activating only the k nearest locations. SDM is content-addressable, noise-robust, and has capacity proportional to the number of hard locations."),
    ("transformer",
     "The Transformer is a neural network architecture introduced in 2017 by Vaswani et al. It uses self-attention mechanisms instead of recurrence, enabling parallel training. Transformers power modern LLMs like GPT, BERT, and T5. They require significant GPU resources to train."),
    ("water",
     "Water is a chemical compound with formula H2O. It covers about 71 percent of the Earth's surface. Water boils at 100 degrees Celsius and freezes at 0 degrees Celsius at sea level. It is essential for all known forms of life."),
    ("sun",
     "The Sun is a star at the center of the Solar System. It is a nearly perfect ball of hot plasma. Its diameter is about 1.39 million kilometers. The Sun accounts for about 99.86 percent of the total mass of the Solar System."),
    ("earth",
     "Earth is the third planet from the Sun and the only known astronomical object to harbor life. It has a radius of approximately 6371 kilometers and a population of over 8 billion humans. Earth's atmosphere is composed primarily of nitrogen and oxygen."),
    ("moon",
     "The Moon is Earth's only natural satellite. It has a diameter of about 3474 kilometers. The Moon orbits Earth at an average distance of 384400 kilometers. It is the only celestial body beyond Earth where humans have set foot."),
    ("ai",
     "Artificial Intelligence (AI) is the simulation of human intelligence in machines. It includes machine learning, deep learning, natural language processing, computer vision, and robotics. Modern AI is largely based on neural networks and transformers."),
]


# ---------------------------------------------------------------------------
# HD RAG store
# ---------------------------------------------------------------------------

@dataclass
class WebDoc:
    """A retrieved web document, encoded as an HD vector."""
    topic: str
    text: str
    vector: HDVector
    source: str = "offline_wiki"
    metadata: Dict[str, Any] = field(default_factory=dict)


class HDRAGStore:
    """HD-based retrieval-augmented generation store.

    Stores web documents as HD vectors in a dedicated SDM. At query time,
    encodes the query as HD, reads from SDM, and returns the most similar doc.
    """

    def __init__(self, encoder: TextEncoder, dim: int = DIM, n_locations: int = 3000, k: int = 15):
        self.encoder = encoder
        self.dim = dim
        self.docs: List[WebDoc] = []
        # Dedicated SDM for documents (separate from the agent's KB)
        self.sdm = SparseDistributedMemory(dim=dim, n_locations=n_locations, k=k)
        # Pre-load the offline wiki
        self._load_offline_wiki()

    def _load_offline_wiki(self) -> None:
        """Load the curated offline wiki into the store."""
        for topic, text in OFFLINE_WIKI:
            self.add_document(topic, text, source="offline_wiki")

    def add_document(self, topic: str, text: str, source: str = "user", metadata: Optional[Dict] = None) -> None:
        """Add a document to the RAG store."""
        # Encode the document text as an HD vector
        vec = self.encoder.encode_text(text)
        # Also bundle in the topic vector for stronger topical signal
        topic_vec = self.encoder.encode_text(topic)
        combined = vec.bundle(topic_vec)
        doc = WebDoc(
            topic=topic,
            text=text,
            vector=combined,
            source=source,
            metadata=metadata or {},
        )
        self.docs.append(doc)
        # Write to SDM: address = combined vector, data = combined vector
        # (so reading the same address returns the same vector)
        self.sdm.write(combined, combined)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[WebDoc, float]]:
        """Retrieve the top_k most similar documents for a query.

        Uses DIRECT cosine similarity between the query HD vector and each
        document's HD vector. The SDM is still used for storage (so writes
        accumulate evidence) but retrieval is direct for precision.
        """
        query_vec = self.encoder.encode_text(query)
        # Direct similarity comparison (more precise than SDM read)
        sims = []
        for doc in self.docs:
            sim = query_vec.similarity(doc.vector)
            sims.append((doc, sim))
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    def retrieve_text(self, query: str, top_k: int = 1) -> Optional[str]:
        """Retrieve the text of the top-1 document for a query."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return None
        return results[0][0].text

    def stats(self) -> Dict[str, Any]:
        return {
            "n_docs": len(self.docs),
            "dim": self.dim,
            "sdm": self.sdm.stats(),
            "sources": list(set(d.source for d in self.docs)),
        }


# ---------------------------------------------------------------------------
# Web fetcher (with graceful fallback)
# ---------------------------------------------------------------------------

class WebFetcher:
    """Fetch web content for a query, with offline fallback.

    Strategy:
      1. Try to fetch from a real web search API (DuckDuckGo HTML or Wikipedia API).
      2. If no internet or fetch fails, fall back to the offline wiki.
    """

    WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    def __init__(self, rag_store: HDRAGStore, timeout: float = 3.0):
        self.rag = rag_store
        self.timeout = timeout
        self._has_internet: Optional[bool] = None

    def has_internet(self) -> bool:
        """Check if we have internet connectivity (cached)."""
        if self._has_internet is not None:
            return self._has_internet
        try:
            # Quick test: try to reach Wikipedia
            req = Request(self.WIKI_API + "Python", headers={"User-Agent": "AETHER/2.0"})
            urlopen(req, timeout=self.timeout).read(100)
            self._has_internet = True
        except Exception:
            self._has_internet = False
        return self._has_internet

    def fetch_wiki(self, topic: str) -> Optional[str]:
        """Fetch a Wikipedia summary for a topic. Returns None on failure."""
        topic_clean = topic.strip().lower().replace(" ", "_")
        url = self.WIKI_API + topic_clean
        try:
            req = Request(url, headers={"User-Agent": "AETHER/2.0 (research prototype)"})
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("type") == "standard":
                    extract = data.get("extract", "")
                    if extract:
                        return extract
        except (URLError, HTTPError, json.JSONDecodeError, OSError):
            return None
        return None

    def fetch_and_store(self, topic: str) -> Tuple[Optional[str], str]:
        """Fetch a topic from the web and add it to the RAG store.

        Returns (text_or_None, source_label).
        """
        # Try web first
        if self.has_internet():
            text = self.fetch_wiki(topic)
            if text:
                self.rag.add_document(topic, text, source="wikipedia")
                return text, "wikipedia"
        # Fall back to offline wiki
        for doc_topic, doc_text in OFFLINE_WIKI:
            if topic.lower() in doc_topic.lower() or doc_topic.lower() in topic.lower():
                return doc_text, "offline_wiki"
        # Last resort: search the RAG store
        results = self.rag.retrieve(topic, top_k=1)
        if results:
            return results[0][0].text, "rag_store"
        return None, "not_found"


# ---------------------------------------------------------------------------
# Tool: web_search (exposed to the agent)
# ---------------------------------------------------------------------------

def tool_web_search(args: str, ctx) -> str:
    """Web search tool: fetch info about a topic and add it to RAG.

    Args is the search query (topic).
    """
    # ctx is a ToolContext from tools.py, but we need access to the RAG store
    # which is on the agent. We access it via ctx.assoc (which is the agent's assoc)
    # but we need a separate RAG store. For now, use a global one.
    global _GLOBAL_RAG
    if _GLOBAL_RAG is None:
        _GLOBAL_RAG = HDRAGStore(ctx.encoder, dim=ctx.encoder.dim)
    query = args.strip()
    if not query:
        return "web_search: empty query"
    fetcher = WebFetcher(_GLOBAL_RAG)
    text, source = fetcher.fetch_and_store(query)
    if text:
        # Truncate for display
        text_short = text if len(text) < 300 else text[:297] + "..."
        return f"[{source}] {text_short}"
    return f"web_search: no results for '{query}'"


def tool_rag_query(args: str, ctx) -> str:
    """RAG query tool: retrieve stored documents matching a query."""
    global _GLOBAL_RAG
    if _GLOBAL_RAG is None:
        _GLOBAL_RAG = HDRAGStore(ctx.encoder, dim=ctx.encoder.dim)
    query = args.strip()
    if not query:
        return "rag_query: empty query"
    results = _GLOBAL_RAG.retrieve(query, top_k=2)
    if not results:
        return "rag_query: nothing found"
    lines = [f"Top {len(results)} results for '{query}':"]
    for i, (doc, sim) in enumerate(results, 1):
        text_short = doc.text if len(doc.text) < 150 else doc.text[:147] + "..."
        lines.append(f"  {i}. [{doc.source}] (sim={sim:.3f}) {text_short}")
    return "\n".join(lines)


def tool_rag_stats(args: str, ctx) -> str:
    """Show RAG store statistics."""
    global _GLOBAL_RAG
    if _GLOBAL_RAG is None:
        _GLOBAL_RAG = HDRAGStore(ctx.encoder, dim=ctx.encoder.dim)
    s = _GLOBAL_RAG.stats()
    return (f"RAG store: {s['n_docs']} docs, dim={s['dim']}, "
            f"sources={s['sources']}, sdm_writes={s['sdm']['total_writes']}")


# Global RAG store (initialized lazily on first use)
_GLOBAL_RAG: Optional[HDRAGStore] = None


def get_global_rag(encoder: TextEncoder) -> HDRAGStore:
    """Get or create the global RAG store."""
    global _GLOBAL_RAG
    if _GLOBAL_RAG is None:
        _GLOBAL_RAG = HDRAGStore(encoder, dim=encoder.dim)
    return _GLOBAL_RAG


def reset_global_rag() -> None:
    """Reset the global RAG store (mainly for testing)."""
    global _GLOBAL_RAG
    _GLOBAL_RAG = None
