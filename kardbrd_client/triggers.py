"""
Trigger mixins for building event-driven Kardbrd agents.

This module provides base classes and specific trigger implementations that agents
can inherit from to respond to different types of Kardbrd board events.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kardbrd_client.agent import BoardSubscription

logger = logging.getLogger("planner")


@dataclass
class TriggerContext:
    """
    Context extracted from a trigger event.

    This contains all relevant information about the activity that triggered
    the agent, structured in a way that's easy to work with.

    Thread Safety:
        This context contains only DATA (strings, primitives, and pure dataclasses).
        No live objects like KardbrdClient or ToolExecutor are passed through.

        The runner creates its own KardbrdClient and ToolExecutor from the
        subscription data, ensuring complete isolation between scheduler
        and worker threads.
    """

    activity_id: str
    board_id: str
    card_id: str
    entity_id: str
    entity_type: str
    action: str
    user: dict[str, Any]
    created_at: str
    description: str
    # Full content fetched from API (not truncated like entity_name)
    comment_content: str | None = None

    # Board subscription data (pure dataclass, no live objects)
    # Contains: api_url, bot_token, ai_provider, ai_model, api_key, etc.
    subscription: BoardSubscription | None = None

    @property
    def comment_id(self) -> str:
        """Alias for entity_id when entity_type is comment."""
        return self.entity_id if self.entity_type == "comment" else None

    @property
    def user_name(self) -> str:
        """Convenience property for user's display name."""
        return self.user.get("display_name", "Unknown")

    @property
    def user_email(self) -> str:
        """Convenience property for user's email."""
        return self.user.get("email", "")


class TriggerMixin(ABC):
    """
    Base mixin for activity-based triggers.

    Agents inherit from this (or specific trigger mixins) to gain trigger
    detection, context extraction, and prompt composition capabilities.
    """

    @abstractmethod
    def should_trigger(self, activity: dict[str, Any]) -> bool:
        """
        Check if this activity should trigger the agent.

        Args:
            activity: Activity dict from the API

        Returns:
            True if this activity matches trigger conditions
        """
        pass

    @abstractmethod
    def extract_context(self, activity: dict[str, Any]) -> TriggerContext:
        """
        Extract relevant context from the activity.

        Args:
            activity: Activity dict from the API

        Returns:
            TriggerContext with structured information
        """
        pass

    @abstractmethod
    def handle_trigger(self, context: TriggerContext) -> None:
        """
        Handle the triggered event.

        Args:
            context: Extracted trigger context
        """
        pass

    def get_system_prompt_addition(self) -> str:
        """
        Return trigger-specific system prompt instructions.

        This explains to the agent HOW it was triggered and what's expected.
        Override this in specific trigger mixins to provide context.

        Returns:
            Additional system prompt text for this trigger type
        """
        return ""

    def get_user_context_message(self, context: TriggerContext) -> str:
        """
        Return trigger-specific user message with context.

        This provides the immediate context for this specific trigger event.
        Override this in specific trigger mixins to format the message.

        Args:
            context: The trigger context

        Returns:
            Formatted user message with trigger context
        """
        return ""

    def get_tool_guidance(self) -> str:
        """
        Return trigger-specific tool usage guidance.

        Different triggers might need different tool patterns.
        Override this to provide trigger-specific tool recommendations.

        Returns:
            Tool usage guidance for this trigger type
        """
        return ""

    def process_activities(self, activities: list[dict[str, Any]]) -> None:
        """
        Process a list of activities and trigger if conditions match.

        This method filters activities and submits matching ones to the
        thread pool executor for concurrent processing.

        Thread Safety:
            Before submitting to the thread pool, this method captures the
            current client, subscription, and tool_executor from the agent
            and attaches them to the context. This ensures the thread has
            all resources it needs without reading instance variables that
            may change.

        Args:
            activities: List of activity dicts from the API
        """
        triggered = [a for a in activities if self.should_trigger(a)]

        for activity in triggered:
            context = self.extract_context(activity)

            # Capture subscription data (pure dataclass, no live objects)
            # Runner will create its own KardbrdClient and ToolExecutor from this data
            context.subscription = getattr(self, "_current_subscription", None)

            # Submit to executor for concurrent processing
            # MultiBaseAgent uses _get_trigger_executor, BaseAgent uses _get_executor
            executor_method = getattr(self, "_get_executor", None) or getattr(self, "_get_trigger_executor", None)
            if not executor_method:
                raise AttributeError(f"{self.__class__.__name__} must provide _get_executor or _get_trigger_executor")

            future = executor_method().submit(self._safe_handle_trigger, context)
            self._futures.append(future)

        # Clean up completed futures
        self._futures = [f for f in self._futures if not f.done()]

    def _safe_handle_trigger(self, context: TriggerContext) -> None:
        """Thread-safe wrapper for handle_trigger with error handling.

        Creates its own client for error posting since runner.py handles
        conversation errors independently.
        """
        try:
            self.handle_trigger(context)
        except Exception as e:
            logger.error(f"Error handling trigger {context.activity_id}: {e}")
            # Create client from context data for error posting
            if context.subscription and context.card_id:
                try:
                    from .client import KardbrdClient

                    client = KardbrdClient(
                        base_url=context.subscription.api_url,
                        token=context.subscription.bot_token,
                    )
                    client.add_comment(
                        context.card_id,
                        f"⚠️ **Agent Error**\n\n{e}\n\n@{context.user_name}",
                    )
                    client.close()
                except Exception:
                    logger.error("Failed to post error comment")


class CommentMentionTrigger(TriggerMixin):
    """
    Trigger on @mentions in comments.

    This mixin triggers when a comment contains a specific keyword (e.g., @planner).
    It automatically filters out bot comments to prevent loops.

    Attributes:
        mention_keyword: The keyword to look for in comments (e.g., "@planner")
        client: KardbrdClient instance for fetching comment content

    Note:
        The implementing class MUST have a `client` attribute (KardbrdClient instance)
        for fetching the full comment content via API.
    """

    mention_keyword: str = None  # Subclass must define this
    client: Any = None  # Subclass must provide KardbrdClient instance

    # Thread-safe cache for fetched comment content (activity_id -> content)
    _comment_cache: dict[str, str] = None
    _cache_lock: threading.Lock = None

    def _get_cache_lock(self) -> threading.Lock:
        """Get or create the cache lock (thread-safe)."""
        if self._cache_lock is None:
            self._cache_lock = threading.Lock()
        return self._cache_lock

    def _get_comment_content(self, activity: dict[str, Any]) -> str | None:
        """
        Fetch full comment content from API (thread-safe).

        Activity only contains truncated content in entity_name (first 50 chars).
        This fetches the complete comment to check for mentions anywhere in text.
        """
        if self.client is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a 'client' attribute "
                "(KardbrdClient instance) for fetching comment content"
            )

        with self._get_cache_lock():
            # Initialize cache if needed
            if self._comment_cache is None:
                self._comment_cache = {}

            activity_id = activity.get("id")
            if activity_id in self._comment_cache:
                return self._comment_cache[activity_id]

        # Fetch outside lock (API call can be slow)
        card_id = activity.get("card_id")
        comment_id = activity.get("entity_id")

        if not card_id or not comment_id:
            return None

        try:
            comment = self.client.get_comment(card_id, comment_id)
            content = comment.get("content", "")

            # Store in cache with lock
            with self._get_cache_lock():
                self._comment_cache[activity_id] = content

            return content
        except Exception:
            return None

    def should_trigger(self, activity: dict[str, Any]) -> bool:
        """Trigger on comment mentions of the keyword."""
        # Must be a comment
        if activity.get("entity_type") != "comment":
            return False

        if activity.get("action") != "commented":
            return False

        # Skip bot comments (prevent loops)
        if activity.get("user", {}).get("is_bot", False):
            logger.debug(f"Skipping bot comment from {activity.get('user', {}).get('display_name')}")
            return False

        # Check for mention keyword
        if not self.mention_keyword:
            raise ValueError(f"{self.__class__.__name__} must define 'mention_keyword' attribute")

        # Fetch full comment content and check for mention
        content = self._get_comment_content(activity)
        if content is None:
            logger.debug(f"Could not fetch comment content for activity {activity.get('id')}")
            return False

        # Case-insensitive mention check
        has_mention = self.mention_keyword.lower() in content.lower()
        logger.debug(f"Comment check: looking for '{self.mention_keyword}' in '{content[:100]}...' = {has_mention}")
        return has_mention

    def extract_context(self, activity: dict[str, Any]) -> TriggerContext:
        """Extract comment mention context."""
        # Get cached comment content (should already be fetched by should_trigger)
        comment_content = self._get_comment_content(activity)

        return TriggerContext(
            activity_id=activity["id"],
            board_id=activity.get("board_id", ""),
            card_id=activity["card_id"],
            entity_id=activity["entity_id"],  # comment_id
            entity_type=activity["entity_type"],
            action=activity["action"],
            user=activity.get("user", {}),
            created_at=activity.get("created_at", ""),
            description=activity.get("description", ""),
            comment_content=comment_content,
        )

    def get_system_prompt_addition(self) -> str:
        """Return prompt instructions for comment mentions."""
        return f"""
## When Triggered by {self.mention_keyword} Mention

You were mentioned by a user who needs your help.

**Your workflow:**
1. Use `get_card_markdown` to understand the full context
2. Take action (create checklist, add todos, update card, etc.)
3. Only comment if you have questions OR to say "Done. @username"

**Keep comments minimal** - the card content is the source of truth.
"""

    def get_user_context_message(self, context: TriggerContext) -> str:
        """Return formatted user message for comment mentions."""
        return f"""Mentioned by {context.user_name} on card {context.card_id}:

"{context.comment_content}"

Use `get_card_markdown` for full context, then take action."""

    def get_tool_guidance(self) -> str:
        """Return tool usage guidance for comment mentions."""
        return """
**Tool flow:**
1. `get_card_markdown(card_id)` - Get context
2. Take action with write tools (create_checklist, add_todo, etc.)
3. `add_comment` only to ask questions or confirm "Done. @username"
"""


class CardAssignmentTrigger(TriggerMixin):
    """
    Trigger when a card is assigned to the bot.

    This mixin triggers when a card is assigned to the bot user, indicating
    that the bot should take ownership or action on that card.

    Attributes:
        bot_user_id: The public_id (UUID) of the bot user
    """

    bot_user_id: str = None  # Subclass must define this

    def should_trigger(self, activity: dict[str, Any]) -> bool:
        """Trigger when card is assigned to this bot."""
        if activity.get("action") != "assigned":
            return False

        if not self.bot_user_id:
            raise ValueError(f"{self.__class__.__name__} must define 'bot_user_id' attribute")

        # Check if assigned to this bot
        assignee_id = activity.get("extra_data", {}).get("assignee_id")
        return assignee_id == self.bot_user_id

    def extract_context(self, activity: dict[str, Any]) -> TriggerContext:
        """Extract assignment context."""
        return TriggerContext(
            activity_id=activity["id"],
            board_id=activity.get("board_id", ""),
            card_id=activity["card_id"],
            entity_id=activity.get("entity_id", ""),
            entity_type=activity.get("entity_type", "card"),
            action=activity["action"],
            user=activity.get("user", {}),
            created_at=activity.get("created_at", ""),
            description=activity.get("description", ""),
        )

    def get_system_prompt_addition(self) -> str:
        """Return prompt instructions for card assignments."""
        return """
## When Triggered by Card Assignment

You were assigned to a card.

**Your workflow:**
1. Use `get_card_markdown` to review the card
2. Create an "AI Planning" checklist if needed
3. Add todo items for actionable steps
4. Only comment if you have questions OR to say "Done. @username"

**Keep comments minimal** - the card content is the source of truth.
"""

    def get_user_context_message(self, context: TriggerContext) -> str:
        """Return formatted user message for card assignments."""
        return f"""Assigned to card {context.card_id} by {context.user_name}.

Use `get_card_markdown` for full context, then take action."""

    def get_tool_guidance(self) -> str:
        """Return tool usage guidance for card assignments."""
        return """
**Tool flow:**
1. `get_card_markdown(card_id)` - Get context
2. `create_checklist(card_id, "AI Planning")` if needed
3. `add_todo` for each actionable step
4. `add_comment` only to ask questions or confirm "Done. @username"
"""


class TodoItemAssignmentTrigger(TriggerMixin):
    """
    Trigger when a todo item is assigned to the bot.

    This mixin triggers when a todo item is assigned to the bot user,
    indicating that the bot should provide refinement or detailed instructions.

    Attributes:
        bot_user_id: The public_id (UUID) of the bot user
        client: KardbrdClient instance for fetching todo item details
    """

    bot_user_id: str = None  # Subclass must define this
    client: Any = None  # Subclass must provide KardbrdClient instance

    def should_trigger(self, activity: dict[str, Any]) -> bool:
        """Trigger when todo item is assigned to this bot."""
        if activity.get("action") != "assigned":
            return False

        if activity.get("entity_type") != "todo_item":
            return False

        if not self.bot_user_id:
            raise ValueError(f"{self.__class__.__name__} must define 'bot_user_id' attribute")

        # Check if assigned to this bot
        assignee_id = activity.get("extra_data", {}).get("assignee_id")
        return assignee_id == self.bot_user_id

    def extract_context(self, activity: dict[str, Any]) -> TriggerContext:
        """Extract todo assignment context with additional todo item info."""
        return TriggerContext(
            activity_id=activity["id"],
            board_id=activity.get("board_id", ""),
            card_id=activity["card_id"],
            entity_id=activity.get("entity_id", ""),  # todo_item public_id
            entity_type=activity.get("entity_type", "todo_item"),
            action=activity["action"],
            user=activity.get("user", {}),
            created_at=activity.get("created_at", ""),
            description=activity.get("description", ""),
        )

    def get_system_prompt_addition(self) -> str:
        """Return prompt instructions for todo item assignments."""
        return """
## When Triggered by Todo Item Assignment

You were assigned to a todo item that needs refinement.

**Your workflow:**
1. Use `get_card_markdown` to understand the full card context
2. Focus on the specific todo item you were assigned to
3. Add a comment with detailed instructions for completing that todo item
4. Reference the specific todo item title in your comment

**Comment format:**
Start your comment with the todo item title, then provide:
- Clear step-by-step instructions
- Any prerequisites or dependencies
- Acceptance criteria if relevant
- Estimated effort/complexity if helpful

**Keep comments focused** - address only the assigned todo item.
"""

    def get_user_context_message(self, context: TriggerContext) -> str:
        """Return formatted user message for todo assignments."""
        return f"""Assigned to todo item on card {context.card_id} by {context.user_name}.

Todo item: "{context.description}" (entity_id: {context.entity_id})

Use `get_card_markdown` to see the full card context, then add a comment with detailed instructions for this specific todo item."""

    def get_tool_guidance(self) -> str:
        """Return tool usage guidance for todo assignments."""
        return """
**Tool flow:**
1. `get_card_markdown(card_id)` - Get full card context including all todo items
2. `add_comment(card_id, content)` - Add detailed instructions for the assigned todo
3. Optionally: `update_todo` to unassign yourself after providing instructions
"""
