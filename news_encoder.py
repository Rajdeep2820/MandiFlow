"""
news_encoder.py  —  MandiFlow v2.0
=====================================
BERT-based semantic encoder for agricultural news headlines.

This module is intentionally kept separate from the main inference
pipeline. It is NOT called during normal simulator.py execution —
news_analyzer.py handles that via Gemini/heuristics.

NewsEncoder is available for:
  - Future fine-tuning of a domain-specific ag-news classifier
  - Offline batch analysis of news corpora
  - Research: comparing semantic similarity of shock events

Key fix from v1:
  v1 ran encoder = NewsEncoder() and a test inference at module level,
  causing a ~30 second BERT load on every `import news_encoder` — which
  happened on every app.py startup via the import chain.

  v2 uses lazy initialisation: the model only loads when you first call
  get_shock_vector(). Nothing runs at import time.

Usage:
    from news_encoder import NewsEncoder
    encoder = NewsEncoder()           # model loads here, once
    vec = encoder.get_shock_vector("Heavy floods hit Nashik onion belt")
    print(vec.shape)                  # torch.Size([1, 768])
"""

import torch


class NewsEncoder:
    """
    Lazy-loaded BERT encoder for agricultural news text.

    The underlying BERT model is downloaded and initialised only on the
    first call to get_shock_vector() — not at import time, and not at
    NewsEncoder() construction time.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self._tokenizer = None
        self._model     = None

    def _ensure_loaded(self):
        """Loads BERT on first use (lazy init)."""
        if self._model is not None:
            return

        try:
            from transformers import BertModel, BertTokenizer
        except ImportError:
            raise ImportError(
                "transformers package not found. "
                "Install with: pip install transformers"
            )

        print(f"📰 Loading BERT encoder ({self.model_name})... ", end="", flush=True)
        self._tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self._model     = BertModel.from_pretrained(self.model_name)
        self._model.eval()
        print("done.")

    def get_shock_vector(self, headline: str) -> torch.Tensor:
        """
        Encodes a news headline into a 768-dimensional semantic vector.

        Uses the [CLS] token embedding from the final BERT layer — this
        is the standard sentence-level representation.

        Args:
            headline: Raw news text (English). Max ~512 tokens.

        Returns:
            torch.Tensor of shape (1, 768)
        """
        self._ensure_loaded()

        inputs = self._tokenizer(
            headline,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        # CLS token = index 0 of the last hidden state
        # Shape: (1, 768)
        return outputs.last_hidden_state[:, 0, :]

    def get_batch_vectors(self, headlines: list) -> torch.Tensor:
        """
        Encodes a list of headlines in one batched forward pass.

        Args:
            headlines: List of news strings.

        Returns:
            torch.Tensor of shape (len(headlines), 768)
        """
        self._ensure_loaded()

        inputs = self._tokenizer(
            headlines,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        return outputs.last_hidden_state[:, 0, :]   # (B, 768)


# ---------------------------------------------------------------------------
# STANDALONE TEST — only runs when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    encoder = NewsEncoder()

    test_headlines = [
        "Heavy rainfall floods Mandsaur Mandi garlic stocks",
        "Truckers strike on NH-48 disrupts onion supply to Delhi",
        "Government imposes export ban on onions to control price rise",
    ]

    print("\nTesting NewsEncoder:")
    for headline in test_headlines:
        vec = encoder.get_shock_vector(headline)
        print(f"  [{vec.shape}]  {headline[:60]}")

    print("\nBatch test:")
    batch = encoder.get_batch_vectors(test_headlines)
    print(f"  Batch shape: {batch.shape}  — expected ({len(test_headlines)}, 768)")
    print("\n✅ NewsEncoder test complete.")