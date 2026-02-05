"""Base agent infrastructure for Kardbrd bots.

This module provides reusable base classes and utilities for creating
Kardbrd board monitoring agents (bots).
"""

import json
import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from apscheduler.executors.pool import ThreadPoolExecutor as APSThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

from .client import KardbrdAPIError, KardbrdClient

logger = logging.getLogger(__name__)

AI_PROVIDERS = {
    "anthropic_api": {
        "name": "Anthropic API (Claude)",
        "default_model": "claude-3-5-sonnet-20241022",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "mistral_vibe": {
        "name": "Mistral Vibe CLI",
        "default_model": "mistral-large-latest",
        "env_key": "MISTRAL_API_KEY",
    },
    "gemini_api": {
        "name": "Gemini API",
        "default_model": "gemini-2.0-flash-exp",
        "env_key": "GEMINI_API_KEY",
    },
}

VALID_AI_PROVIDERS = list(AI_PROVIDERS.keys())


# ============================================================================
# State Management
# ============================================================================


@dataclass
class BoardSubscription:
    """Subscription details for a single board."""

    board_id: str
    api_url: str
    bot_token: str
    agent_name: str
    last_activity_id: str | None = None
    last_activity_time: str | None = None
    last_poll_time: str | None = None
    # Bot-specific fields (for refiner/coder)
    bot_user_id: str | None = None
    # Coder-specific
    ai_provider: str | None = None
    ai_model: str | None = None
    ssh_key_path: str | None = None
    api_key: str | None = None

    def to_dict(self) -> dict:
        """Convert subscription to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BoardSubscription":
        """Create subscription from dictionary."""
        return cls(
            board_id=data.get("board_id", ""),
            api_url=data.get("api_url", ""),
            bot_token=data.get("bot_token", ""),
            agent_name=data.get("agent_name", ""),
            last_activity_id=data.get("last_activity_id"),
            last_activity_time=data.get("last_activity_time"),
            last_poll_time=data.get("last_poll_time"),
            bot_user_id=data.get("bot_user_id"),
            ai_provider=data.get("ai_provider"),
            ai_model=data.get("ai_model"),
            ssh_key_path=data.get("ssh_key_path"),
            api_key=data.get("api_key"),
        )


class DirectoryStateManager:
    """Manages multi-board state with per-board file storage for better concurrency.

    State is stored in a directory with one JSON file per board:
        state_dir/
            board-uuid-1.json
            board-uuid-2.json
            ...

    Each board file contains:
        - Board subscription details
        - Activity tracking (last_activity_id, last_activity_time, last_poll_time)
        - Active tasks (for coder bot, per-board)

    Thread safety: Each board has its own lock, allowing concurrent updates
    to different boards without blocking.
    """

    def __init__(self, state_dir: str):
        """
        Initialize with path to state directory.

        Args:
            state_dir: Path to the state directory
        """
        self.state_dir = Path(state_dir)
        self._board_locks: dict[str, threading.RLock] = {}
        self._locks_lock = threading.RLock()  # Lock for the locks dictionary itself

    def _ensure_directory(self) -> None:
        """Create state directory if it doesn't exist."""
        if not self.state_dir.exists():
            self.state_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created state directory: {self.state_dir}")

    def _get_board_lock(self, board_id: str) -> threading.RLock:
        """Get or create a lock for a specific board (thread-safe)."""
        with self._locks_lock:
            if board_id not in self._board_locks:
                self._board_locks[board_id] = threading.RLock()
            return self._board_locks[board_id]

    def _board_file_path(self, board_id: str) -> Path:
        """Get path to board state file."""
        return self.state_dir / f"{board_id}.json"

    def _atomic_write(self, file_path: Path, data: dict) -> None:
        """Atomically write data to file using temp file + rename."""
        temp_file = file_path.with_suffix(".tmp")
        temp_file.write_text(json.dumps(data, indent=2))
        temp_file.rename(file_path)

    def _load_board_state(self, board_id: str) -> dict | None:
        """Load state for a specific board from its file."""
        file_path = self._board_file_path(board_id)
        if not file_path.exists():
            return None
        try:
            return json.loads(file_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load board state for {board_id}: {e}")
            return None

    def add_subscription(self, subscription: BoardSubscription) -> None:
        """Add or update a board subscription (thread-safe per board)."""
        self._ensure_directory()
        board_id = subscription.board_id
        lock = self._get_board_lock(board_id)

        with lock:
            # Load existing state to preserve active_tasks
            existing = self._load_board_state(board_id)
            data = subscription.to_dict()
            if existing and "active_tasks" in existing:
                data["active_tasks"] = existing["active_tasks"]
            else:
                data["active_tasks"] = {}

            self._atomic_write(self._board_file_path(board_id), data)

    def remove_subscription(self, board_id: str) -> bool:
        """Remove a board subscription by deleting its file. Returns True if removed."""
        lock = self._get_board_lock(board_id)
        with lock:
            file_path = self._board_file_path(board_id)
            if file_path.exists():
                file_path.unlink()
                return True
            return False

    def get_subscription(self, board_id: str) -> BoardSubscription | None:
        """Get a specific board subscription."""
        lock = self._get_board_lock(board_id)
        with lock:
            data = self._load_board_state(board_id)
            if data:
                return BoardSubscription.from_dict(data)
            return None

    def get_all_subscriptions(self) -> dict[str, BoardSubscription]:
        """Get all board subscriptions by scanning the directory."""
        if not self.state_dir.exists():
            return {}

        subscriptions = {}
        for file_path in self.state_dir.glob("*.json"):
            board_id = file_path.stem
            try:
                data = json.loads(file_path.read_text())
                subscriptions[board_id] = BoardSubscription.from_dict(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load subscription from {file_path}: {e}")

        return subscriptions

    def update_board_activity(self, board_id: str, activity_id: str, activity_time: str) -> None:
        """Update last seen activity for a specific board."""
        lock = self._get_board_lock(board_id)
        with lock:
            data = self._load_board_state(board_id)
            if data:
                data["last_activity_id"] = activity_id
                data["last_activity_time"] = activity_time
                data["last_poll_time"] = datetime.now().isoformat()
                self._atomic_write(self._board_file_path(board_id), data)

    def get_since_timestamp(self, board_id: str) -> str | None:
        """Get timestamp to use for 'since' parameter for a specific board."""
        data = self._load_board_state(board_id)
        return data.get("last_activity_time") if data else None

    def has_subscriptions(self) -> bool:
        """Check if any board subscriptions exist."""
        if not self.state_dir.exists():
            return False
        return any(self.state_dir.glob("*.json"))

    def reload(self) -> dict[str, BoardSubscription]:
        """Reload all subscriptions from disk (no caching in directory mode)."""
        return self.get_all_subscriptions()

    # Coder-specific: active tasks management (per-board)
    def get_active_tasks(self, board_id: str) -> dict[str, dict]:
        """Get all active tasks for a specific board."""
        lock = self._get_board_lock(board_id)
        with lock:
            data = self._load_board_state(board_id)
            if data:
                return data.get("active_tasks", {})
            return {}

    def update_active_task(self, board_id: str, task_id: str, task_data: dict) -> None:
        """Update an active task for a specific board."""
        lock = self._get_board_lock(board_id)
        with lock:
            data = self._load_board_state(board_id)
            if data:
                if "active_tasks" not in data:
                    data["active_tasks"] = {}
                data["active_tasks"][task_id] = task_data
                self._atomic_write(self._board_file_path(board_id), data)

    def remove_active_task(self, board_id: str, task_id: str) -> None:
        """Remove an active task from a specific board."""
        lock = self._get_board_lock(board_id)
        with lock:
            data = self._load_board_state(board_id)
            if data and "active_tasks" in data and task_id in data["active_tasks"]:
                del data["active_tasks"][task_id]
                self._atomic_write(self._board_file_path(board_id), data)


def prompt_ai_config(provider: str | None = None, model: str | None = None) -> tuple[str, str, str]:
    """
    Interactively prompt user for AI provider, model, and API key.

    Args:
        provider: Pre-selected provider (from CLI arg), or None to prompt
        model: Pre-selected model (from CLI arg), or None to prompt

    Returns:
        tuple of (provider, model, api_key)
    """
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.table import Table

    console = Console()

    # Provider selection (if not pre-selected)
    if not provider:
        console.print("\n[bold]AI Configuration[/bold]\n")

        console.print("Available AI providers:")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Provider", style="green")
        table.add_column("Description", style="white")

        for i, (code, info) in enumerate(AI_PROVIDERS.items(), 1):
            table.add_row(str(i), code, info["name"])

        console.print(table)

        provider_idx = Prompt.ask(
            "\nSelect provider",
            choices=[str(i) for i in range(1, len(VALID_AI_PROVIDERS) + 1)],
            default="1",
        )
        provider = VALID_AI_PROVIDERS[int(provider_idx) - 1]

    # Validate provider is in pre-defined list
    if provider not in VALID_AI_PROVIDERS:
        raise ValueError(f"Invalid provider '{provider}'. Valid options: {', '.join(VALID_AI_PROVIDERS)}")

    provider_info = AI_PROVIDERS[provider]

    # Model prompt with default from pre-defined list (if not pre-selected)
    if not model:
        model = Prompt.ask(
            "\nModel name",
            default=provider_info["default_model"],
        )

    # API key prompt (always ask, can fall back to env var)
    console.print("\n[yellow]Enter API key for provider:[/yellow]")
    console.print("[dim](Press Enter to use environment variable if set)[/dim]")
    api_key = Prompt.ask("API key", default="", password=False)

    # If empty, try to load from environment
    if not api_key:
        env_key = os.getenv(provider_info["env_key"], "")
        if env_key:
            api_key = env_key
            console.print(f"[green]Using API key from {provider_info['env_key']} environment variable[/green]")

    return provider, model, api_key


# ============================================================================
# Base Agent
# ============================================================================


class BaseAgent(ABC):
    """
    Base class for Kardbrd agents using APScheduler.

    This agent can monitor multiple boards in parallel, with each board
    having its own polling job scheduled independently.

    Subclasses should:
    1. Inherit from this class and any required trigger mixins
    2. Implement abstract methods for bot-specific behavior
    3. Override trigger handling methods from mixins as needed
    """

    def __init__(self, state_manager: DirectoryStateManager, poll_interval: int = 30):
        """Initialize the agent."""
        self.state_manager = state_manager
        self.poll_interval = poll_interval

        # Per-board clients (for trigger detection during polling)
        self._clients: dict[str, KardbrdClient] = {}

        # APScheduler setup
        self.scheduler = BackgroundScheduler(
            executors={"default": APSThreadPoolExecutor(10)},
            job_defaults={
                "coalesce": True,  # If job missed, run once (not multiple times)
                "max_instances": 1,  # Only one poll per board at a time
                "misfire_grace_time": 30,  # Allow 30s delay before skipping
            },
        )

        # Thread pool for concurrent trigger handling (separate from scheduler)
        self._trigger_executor: ThreadPoolExecutor | None = None
        self._futures: list[Future] = []
        self._max_workers = 5

        # Currently active board context (for trigger processing)
        self._current_board_id: str | None = None

    def _get_trigger_executor(self) -> ThreadPoolExecutor:
        """Get or create the trigger thread pool executor."""
        if self._trigger_executor is None:
            self._trigger_executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self._trigger_executor

    def _post_error_comment(
        self,
        card_id: str,
        error_message: str,
        tool_name: str | None = None,
        tool_input: dict | None = None,
        user_name: str | None = None,
    ) -> None:
        """Post error comment to card for user visibility.

        Format: User-friendly message + collapsible technical details.

        Args:
            card_id: The card ID to post the comment to
            error_message: The error message to display
            tool_name: Optional tool name that failed
            tool_input: Optional tool input parameters
            user_name: Optional user name to tag
        """
        if not self.client:
            logger.error("Cannot post error comment: no client available")
            return

        try:
            # User-friendly part
            comment = f"⚠️ **Agent Error**\n\n{error_message}"

            # Tag user if known
            if user_name:
                comment += f"\n\n@{user_name}"

            # Collapsible technical details
            if tool_name or tool_input:
                comment += "\n\n<details>\n<summary>Technical Details</summary>\n\n"
                if tool_name:
                    comment += f"**Tool**: `{tool_name}`\n"
                if tool_input:
                    comment += f"**Input**:\n```json\n{json.dumps(tool_input, indent=2)}\n```\n"
                comment += "</details>"

            self.client.add_comment(card_id, comment)
            logger.info(f"Posted error comment to card {card_id}")
        except Exception as e:
            logger.error(f"Failed to post error comment: {e}")

    @abstractmethod
    def get_base_system_prompt(self) -> str:
        """Get the base system prompt for this bot."""
        pass

    @abstractmethod
    def get_logger_name(self) -> str:
        """Get the logger name for this bot (e.g., 'planner', 'refiner')."""
        pass

    def on_subscription_loaded(self, subscription: BoardSubscription) -> None:  # noqa: B027
        """
        Hook called when a subscription is loaded.

        Subclasses can override to set dynamic attributes (e.g., mention_keyword).

        Args:
            subscription: The board subscription
        """
        pass

    @abstractmethod
    def process_activity(self, activities: list[dict[str, Any]]) -> None:
        """
        Process activity updates for the current board.

        Note: self.client and self.tool_executor are set to the current
        board's resources before this method is called.

        Args:
            activities: List of activity dictionaries
        """
        pass

    def _add_board_job(self, board_id: str, subscription: BoardSubscription) -> None:
        """Add a polling job for a board."""
        job_id = f"poll_{board_id}"

        # Create client for this board (used for trigger detection during polling)
        # Worker threads create their own clients from TriggerContext.subscription
        client = KardbrdClient(
            base_url=subscription.api_url,
            token=subscription.bot_token,
        )
        self._clients[board_id] = client

        # Call hook for subclasses
        self.on_subscription_loaded(subscription)

        # Add the polling job
        self.scheduler.add_job(
            func=self._poll_board,
            trigger="interval",
            seconds=self.poll_interval,
            id=job_id,
            args=[board_id],
            replace_existing=True,
        )
        logger.info(f"Added polling job for board {board_id} (every {self.poll_interval}s)")

    def _remove_board_job(self, board_id: str) -> None:
        """Remove a polling job for a board."""
        job_id = f"poll_{board_id}"

        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed polling job for board {board_id}")

        # Cleanup client
        if board_id in self._clients:
            self._clients[board_id].close()
            del self._clients[board_id]

    def _poll_board(self, board_id: str) -> None:
        """Poll a specific board for activity (called by scheduler)."""
        bot_logger = logging.getLogger(self.get_logger_name())

        subscription = self.state_manager.get_subscription(board_id)
        if not subscription:
            bot_logger.warning(f"No subscription for board {board_id}, removing job")
            self._remove_board_job(board_id)
            return

        client = self._clients.get(board_id)
        if not client:
            bot_logger.error(f"No client for board {board_id}")
            return

        since = self.state_manager.get_since_timestamp(board_id)

        try:
            result = client.get_board_activity(
                board_id=board_id,
                since=since,
                limit=50,
            )
            activities = result.get("activities", [])

            if activities:
                bot_logger.info(f"Found {len(activities)} activities on board {board_id}")
                # Update state with most recent activity
                newest = activities[0]
                self.state_manager.update_board_activity(
                    board_id,
                    activity_id=newest["id"],
                    activity_time=newest["created_at"],
                )

                # Process activities with board context
                self._process_activities_for_board(board_id, activities)
            else:
                bot_logger.debug(f"No new activity on board {board_id}")

        except KardbrdAPIError as e:
            bot_logger.error(f"Failed to poll board {board_id}: {e}")

    def _process_activities_for_board(self, board_id: str, activities: list[dict[str, Any]]) -> None:
        """Process activities for a specific board.

        Sets up the board context (client, subscription) before processing.
        The subscription is captured into TriggerContext for worker threads.
        Worker threads create their own client/executor from subscription data.
        """
        # Set current board context for trigger processing
        self._current_board_id = board_id
        self._current_subscription = self.state_manager.get_subscription(board_id)
        # Client is used for trigger detection (e.g., fetching comment content)
        self.client = self._clients.get(board_id)

        try:
            self.process_activity(activities)
        finally:
            # Clear context (worker threads have subscription data via TriggerContext)
            self._current_board_id = None
            self._current_subscription = None
            self.client = None

    def _sync_subscriptions(self) -> None:
        """Sync scheduler jobs with current subscriptions (called periodically)."""
        bot_logger = logging.getLogger(self.get_logger_name())

        # Force reload state from file
        self.state_manager.reload()
        subscriptions = self.state_manager.get_all_subscriptions()

        current_board_ids = set(subscriptions.keys())
        active_job_ids = {
            job.id.replace("poll_", "") for job in self.scheduler.get_jobs() if job.id.startswith("poll_")
        }

        # Add new subscriptions
        for board_id in current_board_ids - active_job_ids:
            bot_logger.info(f"Detected new subscription for board {board_id}")
            self._add_board_job(board_id, subscriptions[board_id])

        # Remove deleted subscriptions
        for board_id in active_job_ids - current_board_ids:
            bot_logger.info(f"Detected removed subscription for board {board_id}")
            self._remove_board_job(board_id)

    def run_loop(self) -> None:
        """Run the agent with APScheduler-based polling."""
        bot_logger = logging.getLogger(self.get_logger_name())
        bot_logger.info(f"Starting {self.get_logger_name().title()} bot (multi-board)...")

        # Load initial subscriptions
        subscriptions = self.state_manager.get_all_subscriptions()

        if not subscriptions:
            bot_logger.warning("No board subscriptions found")
            bot_logger.info(f"Use '{self.get_logger_name()}-bot sub <setup-url>' to subscribe to a board")
        else:
            bot_logger.info(f"Found {len(subscriptions)} subscription(s)")

        # Initialize jobs for existing subscriptions
        for board_id, subscription in subscriptions.items():
            self._add_board_job(board_id, subscription)

        # Add config sync job (check for subscription changes every 60s)
        self.scheduler.add_job(
            func=self._sync_subscriptions,
            trigger="interval",
            seconds=60,
            id="config_sync",
        )

        # Start scheduler
        self.scheduler.start()
        bot_logger.info(f"Scheduler started (poll interval: {self.poll_interval}s)")

        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            bot_logger.info("Shutting down...")
        finally:
            self.scheduler.shutdown(wait=True)
            # Close all clients
            for client in self._clients.values():
                client.close()
            self._clients.clear()
            # Shutdown trigger executor
            if self._trigger_executor:
                self._trigger_executor.shutdown(wait=True)
            bot_logger.info("Shutdown complete")


# ============================================================================
# CLI Utilities
# ============================================================================


def fetch_setup_url(url: str) -> dict[str, Any]:
    """
    Fetch agent credentials from a setup URL.

    Args:
        url: The setup URL to fetch credentials from

    Returns:
        Dictionary containing:
        - board_id: Board public ID
        - token: Bot authentication token
        - agent_name: Agent name for @mentions
        - bot_user_id: Bot user ID (optional, only for some bots)
        - api_url: API base URL extracted from setup URL

    Raises:
        SystemExit: If fetching credentials fails
    """
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        data = response.json()

        board_id = data["board_id"]
        token = data["token"]
        agent_name = data.get("agent_name", "Agent")
        bot_user_id = data.get("bot_user_id")  # Optional

        # Extract API URL from setup URL
        parsed = urlparse(url)
        api_url = f"{parsed.scheme}://{parsed.netloc}"

        print(f"\n✓ Retrieved credentials for {agent_name}")

        result = {
            "board_id": board_id,
            "token": token,
            "agent_name": agent_name,
            "api_url": api_url,
        }

        if bot_user_id:
            result["bot_user_id"] = bot_user_id

        return result

    except httpx.HTTPError as e:
        # Handle error responses (410 for expired, 400 for invalid)
        if hasattr(e, "response") and e.response is not None:
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", str(e))
                logger.error(f"Failed to retrieve token: {error_msg}")
                print(f"\n✗ Error: {error_msg}\n")
            except Exception:
                logger.error(f"Failed to retrieve token: {e}")
                print(f"\n✗ Failed to retrieve agent credentials: {e}\n")
        else:
            logger.error(f"Failed to retrieve token: {e}")
            print(f"\n✗ Failed to retrieve agent credentials: {e}\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        print("\n✗ Invalid response from server\n")
        sys.exit(1)


def is_setup_url(value: str) -> bool:
    """
    Check if a value is a setup URL.

    Args:
        value: String to check

    Returns:
        True if the value looks like a URL, False otherwise
    """
    return value.startswith("http://") or value.startswith("https://")


def validate_manual_subscription(
    board_id: str,
    token: str | None,
    agent_name: str | None,
    required_fields: list[str] | None = None,
) -> None:
    """
    Validate manual subscription parameters.

    Args:
        board_id: Board ID
        token: Bot token
        agent_name: Agent name
        required_fields: Additional required field names (e.g., ["bot_user_id"])

    Raises:
        SystemExit: If validation fails
    """
    required_fields = required_fields or []

    if not board_id:
        logger.error("Board ID is required")
        _print_subscription_usage()
        sys.exit(1)

    if not token:
        logger.error("Bot token is required when providing board_id directly")
        _print_subscription_usage()
        sys.exit(1)

    if not agent_name:
        logger.error("Agent name is required when providing board_id directly")
        _print_subscription_usage()
        sys.exit(1)


def _print_subscription_usage() -> None:
    """Print subscription command usage."""
    print("\nUsage: <bot>-bot sub <setup-url>")
    print("   or: <bot>-bot sub <board-id> <token> --name <agent-name> [options]\n")


def display_subscription_info(
    board_id: str,
    agent_name: str,
    api_url: str,
    token: str,
    state_file: str,
    extra_info: dict[str, str] | None = None,
    ai_provider: str | None = None,
    ai_model: str | None = None,
) -> None:
    """
    Display subscription confirmation information.

    Args:
        board_id: Board ID
        agent_name: Agent name
        api_url: API URL
        token: Bot token (will be truncated)
        state_file: Path to state file
        extra_info: Optional dictionary of additional info to display (e.g., {"Bot user ID": "xxx"})
        ai_provider: AI provider name (optional)
        ai_model: AI model name (optional)
    """
    print(f"\n✓ Subscribed to board: {board_id}")
    print(f"  Agent name: {agent_name} (responds to @{agent_name})")

    # Display extra info if provided
    if extra_info:
        for key, value in extra_info.items():
            print(f"  {key}: {value}")

    print(f"  API URL: {api_url}")
    print(f"  Token: {token[:8]}...")

    if ai_provider:
        print(f"  AI Provider: {ai_provider}")
    if ai_model:
        print(f"  AI Model: {ai_model}")

    print(f"  Config saved to: {state_file}")
    print("\nRun '<bot>-bot start' to begin monitoring\n")
