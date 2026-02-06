"""
Vector store for ConversaVoice.
Provides text embedding and similarity detection for repetition checking.
"""

import json
import logging
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of a similarity check."""
    text: str
    score: float
    is_repetition: bool


class VectorStore:
    """
    Vector store for semantic similarity detection.

    Uses sentence-transformers for text embeddings and cosine similarity
    to detect when users are repeating themselves.
    """

    def __init__(
        self,
        redis_client,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85
    ):
        """
        Initialize the vector store.

        Args:
            redis_client: RedisClient instance for storage.
            model_name: Sentence transformer model name.
            similarity_threshold: Threshold for repetition detection (0.85 = 85%).
        """
        self.redis_client = redis_client
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._model = None

    def _get_model(self):
        """Lazy initialization of sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded")
        return self._model

    def _vectors_key(self, session_id: str) -> str:
        """Generate Redis key for vectors."""
        return f"vectors:{session_id}"

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text.

        Args:
            text: Input text to embed.

        Returns:
            Numpy array of embedding.
        """
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding

    def store_embedding(self, session_id: str, text: str) -> None:
        """
        Store text and its embedding in Redis.

        Args:
            session_id: Session identifier.
            text: Text to store.
        """
        embedding = self.get_embedding(text)
        key = self._vectors_key(session_id)

        # Store as JSON with text and embedding
        data = {
            "text": text,
            "embedding": embedding.tolist()
        }
        self.redis_client.client.rpush(key, json.dumps(data))

    def get_stored_vectors(self, session_id: str) -> List[Tuple[str, np.ndarray]]:
        """
        Retrieve all stored vectors for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of (text, embedding) tuples.
        """
        key = self._vectors_key(session_id)
        stored = self.redis_client.client.lrange(key, 0, -1)

        vectors = []
        for item in stored:
            data = json.loads(item)
            vectors.append((data["text"], np.array(data["embedding"])))

        return vectors

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Similarity score between 0 and 1.
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def check_repetition(
        self,
        session_id: str,
        text: str
    ) -> SimilarityResult:
        """
        Check if the input text is semantically similar to previous messages.

        Args:
            session_id: Session identifier.
            text: New input text to check.

        Returns:
            SimilarityResult with highest match and repetition flag.
        """
        new_embedding = self.get_embedding(text)
        stored_vectors = self.get_stored_vectors(session_id)

        if not stored_vectors:
            # No previous messages, store this one
            self.store_embedding(session_id, text)
            return SimilarityResult(
                text="",
                score=0.0,
                is_repetition=False
            )

        # Find highest similarity
        max_score = 0.0
        most_similar_text = ""

        for stored_text, stored_embedding in stored_vectors:
            score = self.cosine_similarity(new_embedding, stored_embedding)
            if score > max_score:
                max_score = score
                most_similar_text = stored_text

        # Store the new embedding
        self.store_embedding(session_id, text)

        is_repetition = max_score >= self.similarity_threshold

        if is_repetition:
            logger.info(
                f"Repetition detected (score: {max_score:.2f}): "
                f"'{text[:50]}...' similar to '{most_similar_text[:50]}...'"
            )

        return SimilarityResult(
            text=most_similar_text,
            score=max_score,
            is_repetition=is_repetition
        )

    def clear_vectors(self, session_id: str) -> bool:
        """
        Clear all vectors for a session.

        Args:
            session_id: Session identifier.

        Returns:
            True if cleared successfully.
        """
        key = self._vectors_key(session_id)
        self.redis_client.client.delete(key)
        return True
