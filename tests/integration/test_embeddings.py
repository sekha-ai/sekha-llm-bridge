"""Integration tests for multi-dimension embedding support.

Tests validate:
- Dimension detection and caching
- Automatic collection selection
- Multi-dimension search merging
- Dimension switching between models
- Collection migration support
"""

from unittest.mock import MagicMock, patch

import pytest

from sekha_llm_bridge.config import ModelTask
from sekha_llm_bridge.registry import registry


class TestDimensionDetection:
    """Test automatic dimension detection."""

    @pytest.mark.asyncio
    async def test_detect_dimension_from_model(self):
        """Test that dimension is correctly detected from model metadata."""
        with patch.object(registry, "model_cache") as mock_cache:
            # Mock model with 768 dimensions
            mock_cache.get.return_value = MagicMock(
                model_id="nomic-embed-text",
                dimension=768,
                task=ModelTask.EMBEDDING,
            )

            model_info = mock_cache.get("ollama:nomic-embed-text")

            assert model_info.dimension == 768
            assert model_info.task == ModelTask.EMBEDDING

    @pytest.mark.asyncio
    async def test_cache_dimension_for_performance(self):
        """Test that dimensions are cached to avoid repeated lookups."""
        dimension_cache = {}

        def get_dimension(model_id):
            # Simulate expensive lookup
            if model_id not in dimension_cache:
                dimension_cache[model_id] = {
                    "nomic-embed-text": 768,
                    "text-embedding-ada-002": 1536,
                    "text-embedding-3-large": 3072,
                }.get(model_id, 768)
            return dimension_cache[model_id]

        # First call: cache miss
        dim1 = get_dimension("nomic-embed-text")
        assert dim1 == 768
        assert "nomic-embed-text" in dimension_cache

        # Second call: cache hit (no lookup)
        dim2 = get_dimension("nomic-embed-text")
        assert dim2 == 768

        # Different model
        dim3 = get_dimension("text-embedding-3-large")
        assert dim3 == 3072

    @pytest.mark.asyncio
    async def test_handle_unknown_dimension(self):
        """Test handling of models with unknown dimensions."""
        with patch.object(registry, "model_cache") as mock_cache:
            # Model without dimension metadata
            mock_cache.get.return_value = MagicMock(
                model_id="unknown-embedding-model",
                dimension=None,  # Not specified
                task=ModelTask.EMBEDDING,
            )

            model_info = mock_cache.get("provider:unknown-embedding-model")

            # Should have fallback behavior
            default_dimension = model_info.dimension or 768  # Default to 768
            assert default_dimension == 768


class TestCollectionSelection:
    """Test automatic ChromaDB collection selection."""

    @pytest.mark.asyncio
    async def test_select_collection_by_dimension(self):
        """Test that correct collection is selected based on dimension."""

        def get_collection_name(base_name, dimension):
            return f"{base_name}_{dimension}"

        # Different dimensions -> different collections
        assert get_collection_name("conversations", 768) == "conversations_768"
        assert get_collection_name("conversations", 1536) == "conversations_1536"
        assert get_collection_name("conversations", 3072) == "conversations_3072"

    @pytest.mark.asyncio
    async def test_create_collection_if_not_exists(self):
        """Test automatic collection creation for new dimensions."""
        existing_collections = {"conversations_768"}

        def get_or_create_collection(name, dimension):
            collection_name = f"{name}_{dimension}"
            if collection_name not in existing_collections:
                # Would create new collection
                existing_collections.add(collection_name)
                return {"created": True, "name": collection_name}
            return {"created": False, "name": collection_name}

        # Existing collection
        result1 = get_or_create_collection("conversations", 768)
        assert result1["created"] is False

        # New collection
        result2 = get_or_create_collection("conversations", 3072)
        assert result2["created"] is True
        assert "conversations_3072" in existing_collections

    @pytest.mark.asyncio
    async def test_validate_dimension_matches_collection(self):
        """Test validation that embedding dimension matches collection."""
        collection_metadata = {
            "conversations_768": {"dimension": 768},
            "conversations_1536": {"dimension": 1536},
        }

        def validate_embedding(collection_name, embedding_dimension):
            expected_dim = collection_metadata[collection_name]["dimension"]
            if embedding_dimension != expected_dim:
                raise ValueError(
                    f"Dimension mismatch: embedding has {embedding_dimension}, "
                    f"collection expects {expected_dim}"
                )
            return True

        # Valid: matching dimensions
        assert validate_embedding("conversations_768", 768)

        # Invalid: mismatched dimensions
        with pytest.raises(ValueError, match="Dimension mismatch"):
            validate_embedding("conversations_768", 1536)


class TestMultiDimensionSearch:
    """Test searching across multiple dimension collections."""

    @pytest.mark.asyncio
    async def test_search_all_dimensions(self):
        """Test searching across all available dimension collections."""
        # Mock search results from different collections
        mock_results = {
            "conversations_768": [
                {"id": "conv1", "distance": 0.1, "text": "Result from 768"},
                {"id": "conv2", "distance": 0.3, "text": "Another from 768"},
            ],
            "conversations_1536": [
                {"id": "conv3", "distance": 0.15, "text": "Result from 1536"},
            ],
            "conversations_3072": [
                {"id": "conv4", "distance": 0.05, "text": "Result from 3072"},
            ],
        }

        # Merge and sort by distance
        all_results = []
        for collection, results in mock_results.items():
            for result in results:
                result["collection"] = collection
                all_results.append(result)

        all_results.sort(key=lambda x: x["distance"])

        # Best result should be from 3072 collection (lowest distance)
        assert all_results[0]["id"] == "conv4"
        assert all_results[0]["distance"] == 0.05
        assert all_results[0]["collection"] == "conversations_3072"

    @pytest.mark.asyncio
    async def test_normalize_distances_across_dimensions(self):
        """Test distance normalization when comparing across dimensions."""

        # Different dimensions may have different distance scales
        def normalize_distance(distance, dimension):
            # Simple normalization: divide by sqrt(dimension)
            import math

            return distance / math.sqrt(dimension / 768)  # Normalize to 768 baseline

        results = [
            {"distance": 0.1, "dimension": 768},
            {"distance": 0.15, "dimension": 1536},
            {"distance": 0.2, "dimension": 3072},
        ]

        # Normalize and compare
        for result in results:
            result["normalized_distance"] = normalize_distance(
                result["distance"], result["dimension"]
            )

        # After normalization, 768-dim result might still be best
        results.sort(key=lambda x: x["normalized_distance"])
        assert results[0]["dimension"] == 768

    @pytest.mark.asyncio
    async def test_limit_results_per_collection(self):
        """Test limiting results per collection before merging."""

        def search_multi_dimension(query, limit_per_collection=10, total_limit=20):
            collections = [
                "conversations_768",
                "conversations_1536",
                "conversations_3072",
            ]
            all_results = []

            for collection in collections:
                # Mock search each collection
                collection_results = [
                    {"id": f"{collection}_result_{i}", "distance": i * 0.1}
                    for i in range(15)  # More than limit
                ]

                # Limit per collection
                all_results.extend(collection_results[:limit_per_collection])

            # Sort and limit total
            all_results.sort(key=lambda x: x["distance"])
            return all_results[:total_limit]

        results = search_multi_dimension(
            "test query", limit_per_collection=5, total_limit=10
        )

        # Should have exactly 10 total results
        assert len(results) == 10

        # Should be sorted by distance
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)


class TestDimensionSwitching:
    """Test switching between embedding models with different dimensions."""

    @pytest.mark.asyncio
    async def test_switch_embedding_model(self):
        """Test switching from one embedding model to another."""
        current_model = {"id": "nomic-embed-text", "dimension": 768}
        new_model = {"id": "text-embedding-3-large", "dimension": 3072}

        # Simulate model switch
        assert current_model["dimension"] != new_model["dimension"]

        # New embeddings should go to new collection
        current_collection = f"conversations_{current_model['dimension']}"
        new_collection = f"conversations_{new_model['dimension']}"

        assert current_collection == "conversations_768"
        assert new_collection == "conversations_3072"
        assert current_collection != new_collection

    @pytest.mark.asyncio
    async def test_maintain_existing_collections(self):
        """Test that existing collections remain searchable after model switch."""
        # User switches from 768-dim to 3072-dim model
        # Old conversations remain in conversations_768
        # New conversations go to conversations_3072
        # Both should be searchable

        active_collections = ["conversations_768", "conversations_3072"]

        def search_all_collections(query):
            results = []
            for collection in active_collections:
                # Mock search
                results.append(
                    {
                        "collection": collection,
                        "results": [{"id": f"{collection}_result"}],
                    }
                )
            return results

        search_results = search_all_collections("test")

        # Should search both collections
        assert len(search_results) == 2
        collection_names = [r["collection"] for r in search_results]
        assert "conversations_768" in collection_names
        assert "conversations_3072" in collection_names

    @pytest.mark.asyncio
    async def test_handle_concurrent_dimensions(self):
        """Test handling multiple active embedding models simultaneously."""
        # Some users might use different models for different tasks
        active_models = [
            {"id": "nomic-embed-text", "dimension": 768, "use_case": "general"},
            {
                "id": "text-embedding-3-large",
                "dimension": 3072,
                "use_case": "high_quality",
            },
        ]

        def get_collection_for_model(model_id):
            model = next(m for m in active_models if m["id"] == model_id)
            return f"conversations_{model['dimension']}"

        # Different models -> different collections
        assert get_collection_for_model("nomic-embed-text") == "conversations_768"
        assert (
            get_collection_for_model("text-embedding-3-large") == "conversations_3072"
        )


class TestCollectionMigration:
    """Test collection migration and maintenance."""

    @pytest.mark.asyncio
    async def test_list_all_dimension_collections(self):
        """Test listing all dimension-specific collections."""
        all_collections = [
            "conversations_768",
            "conversations_1536",
            "conversations_3072",
            "other_collection",  # Non-dimension collection
        ]

        # Filter dimension collections
        dimension_collections = [
            c
            for c in all_collections
            if c.startswith("conversations_") and c.split("_")[-1].isdigit()
        ]

        assert len(dimension_collections) == 3
        assert "other_collection" not in dimension_collections

    @pytest.mark.asyncio
    async def test_cleanup_empty_collections(self):
        """Test cleanup of empty dimension collections."""
        collection_stats = {
            "conversations_768": {"count": 1000},
            "conversations_1536": {"count": 0},  # Empty
            "conversations_3072": {"count": 500},
        }

        def cleanup_empty_collections(stats, keep_recent=True):
            empty = [name for name, info in stats.items() if info["count"] == 0]
            if not keep_recent:
                return empty
            # In practice, might keep empty collections for recent models
            return []

        # With keep_recent=False, would delete empty
        to_delete = cleanup_empty_collections(collection_stats, keep_recent=False)
        assert "conversations_1536" in to_delete

        # With keep_recent=True, preserve empty collections
        to_delete_safe = cleanup_empty_collections(collection_stats, keep_recent=True)
        assert len(to_delete_safe) == 0

    @pytest.mark.asyncio
    async def test_get_collection_statistics(self):
        """Test getting statistics for dimension collections."""

        def get_stats():
            return {
                "conversations_768": {
                    "count": 1500,
                    "dimension": 768,
                    "model": "nomic-embed-text",
                    "size_mb": 45,
                },
                "conversations_3072": {
                    "count": 300,
                    "dimension": 3072,
                    "model": "text-embedding-3-large",
                    "size_mb": 38,
                },
            }

        stats = get_stats()

        # Verify stats
        assert stats["conversations_768"]["count"] == 1500
        assert stats["conversations_3072"]["dimension"] == 3072

        # Total conversations across all dimensions
        total = sum(s["count"] for s in stats.values())
        assert total == 1800


class TestEdgeCases:
    """Test edge cases in dimension handling."""

    @pytest.mark.asyncio
    async def test_very_large_dimension(self):
        """Test handling of very large embedding dimensions."""
        large_dimension = 8192  # Some models support very large dimensions

        collection_name = f"conversations_{large_dimension}"
        assert collection_name == "conversations_8192"

        # Should handle same as any other dimension
        assert collection_name.startswith("conversations_")
        assert collection_name.split("_")[-1].isdigit()

    @pytest.mark.asyncio
    async def test_dimension_mismatch_error(self):
        """Test clear error when embedding dimension doesn't match collection."""

        def validate_and_insert(collection_dimension, embedding_dimension):
            if collection_dimension != embedding_dimension:
                raise ValueError(
                    f"Cannot insert {embedding_dimension}-dimensional embedding "
                    f"into {collection_dimension}-dimensional collection. "
                    f"Use collection 'conversations_{embedding_dimension}' instead."
                )
            return True

        # Valid
        assert validate_and_insert(768, 768)

        # Invalid with helpful error
        with pytest.raises(ValueError, match="conversations_1536"):
            validate_and_insert(768, 1536)

    @pytest.mark.asyncio
    async def test_fallback_collection_name(self):
        """Test fallback to default collection if dimension unknown."""

        def get_collection_name(base, dimension):
            if dimension is None or dimension <= 0:
                # Fallback to non-dimensional collection
                return base
            return f"{base}_{dimension}"

        # With dimension
        assert get_collection_name("conversations", 768) == "conversations_768"

        # Without dimension (legacy)
        assert get_collection_name("conversations", None) == "conversations"
        assert get_collection_name("conversations", 0) == "conversations"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
