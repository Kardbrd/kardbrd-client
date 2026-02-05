"""Tests for TriggerContext data validation and properties."""

from kardbrd_client.triggers import TriggerContext


class TestTriggerContextCreation:
    """Test TriggerContext creation and basic functionality."""

    def test_creates_with_minimal_fields(self):
        """TriggerContext creates successfully with required fields."""
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="entity-1",
            entity_type="comment",
            action="commented",
            user={"display_name": "Test User"},
            created_at="2024-01-01T00:00:00Z",
            description="Test description",
        )
        assert context.activity_id == "act-1"
        assert context.board_id == "board-1"
        assert context.card_id == "card-1"
        assert context.entity_type == "comment"

    def test_creates_with_subscription(self):
        """TriggerContext creates successfully with subscription data."""
        from dataclasses import dataclass

        @dataclass
        class MockSubscription:
            board_id: str = "board-123"
            api_url: str = "http://api.example.com"
            bot_token: str = "token-123"
            ai_provider: str = "anthropic_api"
            ai_model: str = "claude-3"
            api_key: str = "sk-test-key"

        subscription = MockSubscription()
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="entity-1",
            entity_type="comment",
            action="commented",
            user={"display_name": "Test User"},
            created_at="2024-01-01T00:00:00Z",
            description="Test",
            subscription=subscription,
        )
        assert context.subscription is not None
        assert context.subscription.api_url == "http://api.example.com"
        assert context.subscription.bot_token == "token-123"
        assert context.subscription.ai_provider == "anthropic_api"
        assert context.subscription.api_key == "sk-test-key"


class TestTriggerContextProperties:
    """Test TriggerContext computed properties."""

    def test_comment_id_returns_entity_id_for_comment(self):
        """comment_id returns entity_id when entity_type is comment."""
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="comment-123",
            entity_type="comment",
            action="commented",
            user={},
            created_at="",
            description="",
        )
        assert context.comment_id == "comment-123"

    def test_comment_id_returns_none_for_non_comment(self):
        """comment_id returns None when entity_type is not comment."""
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="card-123",
            entity_type="card",
            action="created",
            user={},
            created_at="",
            description="",
        )
        assert context.comment_id is None

    def test_user_name_returns_display_name(self):
        """user_name returns the display_name from user dict."""
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="entity-1",
            entity_type="comment",
            action="commented",
            user={"display_name": "John Doe", "email": "john@example.com"},
            created_at="",
            description="",
        )
        assert context.user_name == "John Doe"

    def test_user_name_returns_unknown_when_missing(self):
        """user_name returns 'Unknown' when display_name is missing."""
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="entity-1",
            entity_type="comment",
            action="commented",
            user={},
            created_at="",
            description="",
        )
        assert context.user_name == "Unknown"

    def test_user_email_returns_email(self):
        """user_email returns the email from user dict."""
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="entity-1",
            entity_type="comment",
            action="commented",
            user={"display_name": "John", "email": "john@example.com"},
            created_at="",
            description="",
        )
        assert context.user_email == "john@example.com"

    def test_user_email_returns_empty_when_missing(self):
        """user_email returns empty string when email is missing."""
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="entity-1",
            entity_type="comment",
            action="commented",
            user={"display_name": "John"},
            created_at="",
            description="",
        )
        assert context.user_email == ""


class TestTriggerContextDataOnly:
    """Test that TriggerContext is DATA only - no live objects."""

    def test_context_has_no_client_field(self):
        """TriggerContext should not have a client field."""
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="entity-1",
            entity_type="comment",
            action="commented",
            user={},
            created_at="",
            description="",
        )
        # client field should not exist
        assert not hasattr(context, "client")

    def test_context_has_no_tool_executor_field(self):
        """TriggerContext should not have a tool_executor field."""
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="entity-1",
            entity_type="comment",
            action="commented",
            user={},
            created_at="",
            description="",
        )
        # tool_executor field should not exist
        assert not hasattr(context, "tool_executor")

    def test_subscription_is_dataclass_not_live_object(self):
        """subscription field contains data only, not live connections."""
        from dataclasses import dataclass, is_dataclass

        @dataclass
        class MockSubscription:
            api_url: str = "http://api.example.com"
            bot_token: str = "token"

        subscription = MockSubscription()
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="entity-1",
            entity_type="comment",
            action="commented",
            user={},
            created_at="",
            description="",
            subscription=subscription,
        )

        # subscription should be a dataclass (data only)
        assert is_dataclass(context.subscription)
        # It should have no methods that suggest live connections
        assert not hasattr(context.subscription, "connect")
        assert not hasattr(context.subscription, "close")
        assert not hasattr(context.subscription, "execute")


class TestTriggerContextOptionalFields:
    """Test optional fields on TriggerContext."""

    def test_comment_content_defaults_to_none(self):
        """comment_content defaults to None."""
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="entity-1",
            entity_type="comment",
            action="commented",
            user={},
            created_at="",
            description="",
        )
        assert context.comment_content is None

    def test_comment_content_can_be_set(self):
        """comment_content can be set to a string."""
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="entity-1",
            entity_type="comment",
            action="commented",
            user={},
            created_at="",
            description="",
            comment_content="Hello @planner, please help!",
        )
        assert context.comment_content == "Hello @planner, please help!"

    def test_subscription_defaults_to_none(self):
        """subscription defaults to None."""
        context = TriggerContext(
            activity_id="act-1",
            board_id="board-1",
            card_id="card-1",
            entity_id="entity-1",
            entity_type="comment",
            action="commented",
            user={},
            created_at="",
            description="",
        )
        assert context.subscription is None
