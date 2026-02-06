"""Comprehensive tests for Pydantic request and response models."""

import pytest
from pydantic import ValidationError

from sekha_llm_bridge.models.requests import (
    EmbedRequest,
    SummarizeRequest,
    ExtractRequest,
    ScoreRequest,
    ChatMessage,
    ChatCompletionRequest,
    Message,
    EmbeddingRequest,
)


class TestEmbedRequest:
    """Test EmbedRequest model."""

    def test_embed_request_basic(self):
        """Test basic EmbedRequest creation."""
        request = EmbedRequest(text="Hello world")
        assert request.text == "Hello world"
        assert request.model is None

    def test_embed_request_with_model(self):
        """Test EmbedRequest with custom model."""
        request = EmbedRequest(text="Test", model="custom-model")
        assert request.text == "Test"
        assert request.model == "custom-model"

    def test_embed_request_missing_text(self):
        """Test EmbedRequest requires text field."""
        with pytest.raises(ValidationError):
            EmbedRequest()

    def test_embed_request_empty_text(self):
        """Test EmbedRequest with empty text."""
        request = EmbedRequest(text="")
        assert request.text == ""

    def test_embed_request_unicode_text(self):
        """Test EmbedRequest with unicode text."""
        request = EmbedRequest(text="Hello 世界 🌍")
        assert "世界" in request.text

    def test_embed_request_json_serialization(self):
        """Test EmbedRequest JSON serialization."""
        request = EmbedRequest(text="Test")
        json_data = request.model_dump()
        assert json_data["text"] == "Test"


class TestSummarizeRequest:
    """Test SummarizeRequest model."""

    def test_summarize_request_basic(self):
        """Test basic SummarizeRequest creation."""
        request = SummarizeRequest(messages=["msg1", "msg2"])
        assert len(request.messages) == 2
        assert request.level == "daily"  # Default
        assert request.model is None

    def test_summarize_request_with_level(self):
        """Test SummarizeRequest with custom level."""
        request = SummarizeRequest(messages=["test"], level="weekly")
        assert request.level == "weekly"

    def test_summarize_request_with_model(self):
        """Test SummarizeRequest with custom model."""
        request = SummarizeRequest(messages=["test"], model="gpt-4o")
        assert request.model == "gpt-4o"

    def test_summarize_request_missing_messages(self):
        """Test SummarizeRequest requires messages field."""
        with pytest.raises(ValidationError):
            SummarizeRequest()

    def test_summarize_request_empty_messages(self):
        """Test SummarizeRequest with empty messages list."""
        request = SummarizeRequest(messages=[])
        assert request.messages == []

    def test_summarize_request_single_message(self):
        """Test SummarizeRequest with single message."""
        request = SummarizeRequest(messages=["single"])
        assert len(request.messages) == 1

    def test_summarize_request_many_messages(self):
        """Test SummarizeRequest with many messages."""
        messages = [f"msg{i}" for i in range(100)]
        request = SummarizeRequest(messages=messages)
        assert len(request.messages) == 100


class TestExtractRequest:
    """Test ExtractRequest model."""

    def test_extract_request_basic(self):
        """Test basic ExtractRequest creation."""
        request = ExtractRequest(text="Extract entities from this")
        assert request.text == "Extract entities from this"
        assert request.model is None

    def test_extract_request_with_model(self):
        """Test ExtractRequest with custom model."""
        request = ExtractRequest(text="Test", model="custom-model")
        assert request.model == "custom-model"

    def test_extract_request_missing_text(self):
        """Test ExtractRequest requires text field."""
        with pytest.raises(ValidationError):
            ExtractRequest()

    def test_extract_request_long_text(self):
        """Test ExtractRequest with long text."""
        long_text = "word " * 1000
        request = ExtractRequest(text=long_text)
        assert len(request.text) > 1000


class TestScoreRequest:
    """Test ScoreRequest model."""

    def test_score_request_basic(self):
        """Test basic ScoreRequest creation."""
        request = ScoreRequest(conversation="Important conversation")
        assert request.conversation == "Important conversation"
        assert request.model is None

    def test_score_request_with_model(self):
        """Test ScoreRequest with custom model."""
        request = ScoreRequest(conversation="Test", model="gpt-4o")
        assert request.model == "gpt-4o"

    def test_score_request_missing_conversation(self):
        """Test ScoreRequest requires conversation field."""
        with pytest.raises(ValidationError):
            ScoreRequest()

    def test_score_request_empty_conversation(self):
        """Test ScoreRequest with empty conversation."""
        request = ScoreRequest(conversation="")
        assert request.conversation == ""


class TestChatMessage:
    """Test ChatMessage model."""

    def test_chat_message_basic(self):
        """Test basic ChatMessage creation."""
        message = ChatMessage(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"

    def test_chat_message_system_role(self):
        """Test ChatMessage with system role."""
        message = ChatMessage(role="system", content="You are helpful")
        assert message.role == "system"

    def test_chat_message_assistant_role(self):
        """Test ChatMessage with assistant role."""
        message = ChatMessage(role="assistant", content="I can help")
        assert message.role == "assistant"

    def test_chat_message_missing_role(self):
        """Test ChatMessage requires role field."""
        with pytest.raises(ValidationError):
            ChatMessage(content="Test")

    def test_chat_message_missing_content(self):
        """Test ChatMessage requires content field."""
        with pytest.raises(ValidationError):
            ChatMessage(role="user")

    def test_chat_message_empty_content(self):
        """Test ChatMessage with empty content."""
        message = ChatMessage(role="user", content="")
        assert message.content == ""

    def test_chat_message_long_content(self):
        """Test ChatMessage with long content."""
        long_content = "word " * 10000
        message = ChatMessage(role="user", content=long_content)
        assert len(message.content) > 10000


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_chat_completion_request_basic(self):
        """Test basic ChatCompletionRequest creation."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatCompletionRequest(messages=messages)
        assert len(request.messages) == 1
        assert request.model is None
        assert request.temperature == 0.7  # Default
        assert request.max_tokens == 2000  # Default
        assert request.stream is False  # Default

    def test_chat_completion_request_with_model(self):
        """Test ChatCompletionRequest with custom model."""
        messages = [ChatMessage(role="user", content="Test")]
        request = ChatCompletionRequest(messages=messages, model="gpt-4o")
        assert request.model == "gpt-4o"

    def test_chat_completion_request_with_temperature(self):
        """Test ChatCompletionRequest with custom temperature."""
        messages = [ChatMessage(role="user", content="Test")]
        request = ChatCompletionRequest(messages=messages, temperature=0.5)
        assert request.temperature == 0.5

    def test_chat_completion_request_temperature_validation(self):
        """Test ChatCompletionRequest temperature must be between 0 and 2."""
        messages = [ChatMessage(role="user", content="Test")]

        # Valid temperatures
        request = ChatCompletionRequest(messages=messages, temperature=0.0)
        assert request.temperature == 0.0

        request = ChatCompletionRequest(messages=messages, temperature=2.0)
        assert request.temperature == 2.0

        # Invalid temperatures
        with pytest.raises(ValidationError):
            ChatCompletionRequest(messages=messages, temperature=-0.1)

        with pytest.raises(ValidationError):
            ChatCompletionRequest(messages=messages, temperature=2.1)

    def test_chat_completion_request_with_max_tokens(self):
        """Test ChatCompletionRequest with custom max_tokens."""
        messages = [ChatMessage(role="user", content="Test")]
        request = ChatCompletionRequest(messages=messages, max_tokens=100)
        assert request.max_tokens == 100

    def test_chat_completion_request_max_tokens_validation(self):
        """Test ChatCompletionRequest max_tokens must be >= 1."""
        messages = [ChatMessage(role="user", content="Test")]

        # Valid
        request = ChatCompletionRequest(messages=messages, max_tokens=1)
        assert request.max_tokens == 1

        # Invalid
        with pytest.raises(ValidationError):
            ChatCompletionRequest(messages=messages, max_tokens=0)

        with pytest.raises(ValidationError):
            ChatCompletionRequest(messages=messages, max_tokens=-1)

    def test_chat_completion_request_with_streaming(self):
        """Test ChatCompletionRequest with streaming enabled."""
        messages = [ChatMessage(role="user", content="Test")]
        request = ChatCompletionRequest(messages=messages, stream=True)
        assert request.stream is True

    def test_chat_completion_request_missing_messages(self):
        """Test ChatCompletionRequest requires messages field."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest()

    def test_chat_completion_request_empty_messages(self):
        """Test ChatCompletionRequest with empty messages list."""
        request = ChatCompletionRequest(messages=[])
        assert request.messages == []

    def test_chat_completion_request_multiple_messages(self):
        """Test ChatCompletionRequest with multiple messages."""
        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="How are you?"),
        ]
        request = ChatCompletionRequest(messages=messages)
        assert len(request.messages) == 4

    def test_chat_completion_request_conversation_flow(self):
        """Test ChatCompletionRequest with realistic conversation flow."""
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant"),
            ChatMessage(role="user", content="What's the weather?"),
            ChatMessage(role="assistant", content="I need your location"),
            ChatMessage(role="user", content="New York"),
        ]
        request = ChatCompletionRequest(
            messages=messages, model="gpt-4o", temperature=0.7, max_tokens=500
        )

        assert len(request.messages) == 4
        assert request.messages[0].role == "system"
        assert request.messages[-1].role == "user"


class TestMessage:
    """Test Message model (used in providers)."""

    def test_message_basic(self):
        """Test basic Message creation."""
        message = Message(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"

    def test_message_all_roles(self):
        """Test Message with all valid roles."""
        for role in ["system", "user", "assistant", "function", "tool"]:
            message = Message(role=role, content="Test")
            assert message.role == role

    def test_message_missing_fields(self):
        """Test Message requires role and content."""
        with pytest.raises(ValidationError):
            Message(role="user")

        with pytest.raises(ValidationError):
            Message(content="Test")


class TestEmbeddingRequest:
    """Test EmbeddingRequest model (used in providers)."""

    def test_embedding_request_string_input(self):
        """Test EmbeddingRequest with string input."""
        request = EmbeddingRequest(model="text-embedding-3-small", input="Hello")
        assert request.model == "text-embedding-3-small"
        assert request.input == "Hello"

    def test_embedding_request_list_input(self):
        """Test EmbeddingRequest with list input."""
        request = EmbeddingRequest(
            model="text-embedding-3-small", input=["Hello", "World"]
        )
        assert isinstance(request.input, list)
        assert len(request.input) == 2

    def test_embedding_request_with_dimensions(self):
        """Test EmbeddingRequest with dimensions parameter."""
        request = EmbeddingRequest(
            model="text-embedding-3-small", input="Test", dimensions=256
        )
        assert request.dimensions == 256

    def test_embedding_request_missing_model(self):
        """Test EmbeddingRequest requires model."""
        with pytest.raises(ValidationError):
            EmbeddingRequest(input="Test")

    def test_embedding_request_missing_input(self):
        """Test EmbeddingRequest requires input."""
        with pytest.raises(ValidationError):
            EmbeddingRequest(model="test-model")


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_chat_message_to_dict(self):
        """Test ChatMessage serialization to dict."""
        message = ChatMessage(role="user", content="Hello")
        data = message.model_dump()

        assert data["role"] == "user"
        assert data["content"] == "Hello"

    def test_chat_completion_request_to_dict(self):
        """Test ChatCompletionRequest serialization."""
        messages = [ChatMessage(role="user", content="Test")]
        request = ChatCompletionRequest(
            messages=messages, model="gpt-4o", temperature=0.5
        )
        data = request.model_dump()

        assert data["model"] == "gpt-4o"
        assert data["temperature"] == 0.5
        assert len(data["messages"]) == 1

    def test_chat_message_from_dict(self):
        """Test ChatMessage deserialization from dict."""
        data = {"role": "user", "content": "Hello"}
        message = ChatMessage(**data)

        assert message.role == "user"
        assert message.content == "Hello"

    def test_chat_completion_request_from_dict(self):
        """Test ChatCompletionRequest deserialization."""
        data = {
            "messages": [{"role": "user", "content": "Test"}],
            "model": "gpt-4o",
            "temperature": 0.5,
        }
        request = ChatCompletionRequest(**data)

        assert request.model == "gpt-4o"
        assert request.temperature == 0.5
        assert len(request.messages) == 1


class TestModelValidation:
    """Test comprehensive model validation scenarios."""

    def test_invalid_field_types(self):
        """Test models reject invalid field types."""
        # String where list expected
        with pytest.raises(ValidationError):
            SummarizeRequest(messages="not a list")

        # Int where string expected
        with pytest.raises(ValidationError):
            EmbedRequest(text=12345)

    def test_extra_fields_ignored(self):
        """Test models ignore extra fields."""
        # Pydantic by default ignores extra fields
        request = EmbedRequest(text="Test", extra_field="ignored")
        assert request.text == "Test"
        assert not hasattr(request, "extra_field")

    def test_model_immutability(self):
        """Test models can be modified after creation."""
        message = ChatMessage(role="user", content="Original")
        # Pydantic models are mutable by default
        message.content = "Modified"
        assert message.content == "Modified"

    def test_nested_validation(self):
        """Test nested model validation."""
        # Invalid nested message
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[{"role": "user"}]  # Missing content
            )

        # Valid nested message
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Test"}]
        )
        assert len(request.messages) == 1

    def test_optional_fields(self):
        """Test optional fields can be None."""
        request = EmbedRequest(text="Test", model=None)
        assert request.model is None

        request = EmbedRequest(text="Test")
        assert request.model is None
