"""Embedding generation service"""

from typing import List, Optional
import logging

from sekha_llm_bridge.utils.llm_client import llm_client
from sekha_llm_bridge.models.requests import EmbedRequest
from sekha_llm_bridge.models.responses import EmbedResponse

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings"""

    async def generate_embedding(self, request: EmbedRequest) -> EmbedResponse:
        """Generate embedding for text"""
        logger.info(f"Generating embedding for text (length: {len(request.text)})")

        try:
            embedding = await llm_client.generate_embedding(
                text=request.text, model=request.model
            )

            # Estimate tokens (rough: 1 token ≈ 4 chars)
            tokens_used = len(request.text) // 4

            return EmbedResponse(
                embedding=embedding,
                model=request.model or "default",
                dimension=len(embedding),
                tokens_used=tokens_used,  # ADDED: Required field
            )

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def batch_generate_embeddings(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        logger.info(f"Generating batch embeddings for {len(texts)} texts")

        embeddings = []
        for text in texts:
            embedding = await llm_client.generate_embedding(text, model)
            embeddings.append(embedding)

        return embeddings


embedding_service = EmbeddingService()
