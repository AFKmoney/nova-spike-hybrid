"""
BPE Tokenizer — Byte Pair Encoding léger pour SPIKE/NOVA.

Le tokenizer word-level actuel a une limite: chaque mot inconnu est un
nouveau token, ce qui sature vite le vocabulaire.

BPE résout ce problème en apprenant des sous-mots:
  - Initialise avec tous les caractères uniques
  - Itérativement, trouve la paire de tokens la plus fréquente et la fusionne
  - Après N fusions, on a un vocab de N + |chars| tokens

Avantages:
  - Couverture infinie (mots inconnus = combinaison de sous-mots)
  - Vocab fixe (pas de saturation)
  - Meilleure généralisation entre mots apparentés (chat/chats/chaton)

Usage:
    bpe = BPETokenizer(vocab_size=500)
    bpe.train(corpus_text)
    tokens = bpe.encode("bonjour le chat")
    text = bpe.decode(tokens)
"""

from __future__ import annotations
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _word_splits(text: str) -> list[list[str]]:
    """Tokenise en mots puis split en caractères avec marqueur de fin."""
    out = []
    for w in _WORD_RE.findall(text.lower()):
        # Marqueur de fin de mot pour distinguer les sous-mots internes
        # des sous-mots en fin de mot
        out.append(list(w) + ["</w>"])
    return out


@dataclass
class BPETokenizer:
    """
    BPE tokenizer avec vocab fixe.

    Attributes:
        vocab_size: taille cible du vocabulaire
        vocab: dict id -> token (string)
        merges: liste des fusions apprises (dans l'ordre)
    """
    vocab_size: int = 500
    vocab: dict[int, str] = field(init=False, default_factory=dict)
    merges: list[tuple[str, str]] = field(init=False, default_factory=list)
    token2id: dict[str, int] = field(init=False, default_factory=dict)
    _trained: bool = False

    # ---------------------------------------------------------------- #
    # Training
    # ---------------------------------------------------------------- #
    def train(self, corpus: str, verbose: bool = False) -> None:
        """
        Entraîne le BPE sur un corpus.

        Args:
            corpus: texte d'entraînement
            verbose: affiche les fusions
        """
        # Splits initiaux: mots -> caractères + </w>
        word_splits = _word_splits(corpus)
        # Compte les fréquences de mots
        word_freq = Counter()
        for split in word_splits:
            word_freq[tuple(split)] += 1

        # Vocab initial: tous les caractères uniques
        chars = set()
        for split in word_splits:
            chars.update(split)
        chars.add("</w>")
        # Vocab initial
        self.vocab = {i: c for i, c in enumerate(sorted(chars))}
        self.token2id = {c: i for i, c in self.vocab.items()}

        # Compte combien de fusions on peut faire
        n_merges = self.vocab_size - len(self.vocab)
        if n_merges <= 0:
            self._trained = True
            return

        # Représentation des mots comme tuples de tokens
        word_tokens = {w: list(w) for w in word_freq}

        # Boucle BPE
        self.merges = []
        for merge_idx in range(n_merges):
            # Compte les paires de tokens adjacentes
            pair_counts = Counter()
            for word_tuple, freq in word_freq.items():
                tokens = word_tokens[word_tuple]
                for i in range(len(tokens) - 1):
                    pair_counts[(tokens[i], tokens[i+1])] += freq

            if not pair_counts:
                break

            # Trouve la paire la plus fréquente
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            # Nouveau token fusionné
            new_token = best_pair[0] + best_pair[1]
            new_id = len(self.vocab)
            self.vocab[new_id] = new_token
            self.token2id[new_token] = new_id
            self.merges.append(best_pair)

            # Applique la fusion à tous les mots
            for word_tuple in word_tokens:
                tokens = word_tokens[word_tuple]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_tokens[word_tuple] = new_tokens

            if verbose and (merge_idx < 10 or merge_idx % 50 == 0):
                print(f"  merge {merge_idx}: {best_pair} -> {new_token!r} (count={best_count})")

        self._trained = True

    # ---------------------------------------------------------------- #
    # Encoding
    # ---------------------------------------------------------------- #
    def encode(self, text: str) -> list[int]:
        """Encode un texte en liste d'IDs BPE."""
        if not self._trained:
            raise RuntimeError("BPE not trained")

        result = []
        for word in _WORD_RE.findall(text.lower()):
            tokens = self._encode_word(word + "</w>")
            for t in tokens:
                if t in self.token2id:
                    result.append(self.token2id[t])
                else:
                    # Fallback: caractère par caractère
                    for c in t:
                        if c in self.token2id:
                            result.append(self.token2id[c])
        return result

    def _encode_word(self, word: str) -> list[str]:
        """Encode un mot en appliquant les fusions dans l'ordre."""
        tokens = list(word)
        for a, b in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    # ---------------------------------------------------------------- #
    # Decoding
    # ---------------------------------------------------------------- #
    def decode(self, ids: list[int]) -> str:
        """Décode une liste d'IDs en texte."""
        out = []
        for i in ids:
            if i in self.vocab:
                token = self.vocab[i]
                # Retire le marqueur </w> et remplace par espace
                if token.endswith("</w>"):
                    out.append(token[:-4])
                    out.append(" ")
                else:
                    out.append(token)
        text = "".join(out)
        # Nettoie les espaces avant ponctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        return text.strip()

    # ---------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ---------------------------------------------------------------- #
    # Persistance
    # ---------------------------------------------------------------- #
    def save(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "vocab": self.vocab,
                "merges": self.merges,
            }, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.vocab = {int(k): v for k, v in data["vocab"].items()}
        self.merges = [tuple(m) for m in data["merges"]]
        self.token2id = {v: int(k) for k, v in self.vocab.items()}
        self._trained = True


# ---------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Test BPE Tokenizer\n")

    corpus = """
    le chat dort sur le tapis. le chien mange une pomme.
    paris est la capitale de la france. la terre tourne autour du soleil.
    l'eau est composée d'hydrogène et d'oxygène.
    le chat est un animal. le chien est un animal.
    bonjour le monde, comment vas-tu aujourd'hui?
    calcule deux plus deux. que vaut la racine carrée de cent quarante-quatre?
    python est un langage de programmation.
    """

    bpe = BPETokenizer(vocab_size=200)
    bpe.train(corpus, verbose=True)

    print(f"\nVocab final: {len(bpe)} tokens")
    print(f"Merges: {len(bpe.merges)}")

    tests = [
        "le chat",
        "le chien",
        "chaton",  # mot inconnu — doit être décomposé
        "bonjour",
        "calcule deux plus deux",
        "Zorglub",  # mot totalement inconnu
    ]
    for text in tests:
        ids = bpe.encode(text)
        decoded = bpe.decode(ids)
        tokens = [bpe.vocab[i] for i in ids]
        print(f"\n{text!r}")
        print(f"  → IDs: {ids}")
        print(f"  → tokens: {tokens}")
        print(f"  → decoded: {decoded!r}")
