"""
Kardbrd Client - Python client library for the Kardbrd Board API.
"""

from .agent import (
    AI_PROVIDERS,
    VALID_AI_PROVIDERS,
    BaseAgent,
    BoardSubscription,
    DirectoryStateManager,
    display_subscription_info,
    fetch_setup_url,
    is_setup_url,
    prompt_ai_config,
    validate_manual_subscription,
)
from .client import KardbrdAPIError, KardbrdClient
from .logging_config import configure_logging
from .tools import TOOLS, ToolExecutor
from .triggers import (
    CardAssignmentTrigger,
    CommentMentionTrigger,
    TodoItemAssignmentTrigger,
    TriggerContext,
    TriggerMixin,
)
from .websocket_agent import OnboardingEvent, WebSocketAgentConnection

__all__ = [
    "KardbrdClient",
    "KardbrdAPIError",
    "TOOLS",
    "ToolExecutor",
    "TriggerContext",
    "TriggerMixin",
    "CommentMentionTrigger",
    "CardAssignmentTrigger",
    "TodoItemAssignmentTrigger",
    "BaseAgent",
    "BoardSubscription",
    "DirectoryStateManager",
    "AI_PROVIDERS",
    "VALID_AI_PROVIDERS",
    "prompt_ai_config",
    "fetch_setup_url",
    "is_setup_url",
    "validate_manual_subscription",
    "display_subscription_info",
    "configure_logging",
    "WebSocketAgentConnection",
    "OnboardingEvent",
]
__version__ = "1.0.0"
