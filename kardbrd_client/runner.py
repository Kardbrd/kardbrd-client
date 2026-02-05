"""Standalone runner functions for AI conversations.

These functions are completely independent of BaseAgent.
They receive TriggerContext (data only) and create their own resources.

IMPORTANT: Do NOT import anything from agent.py to avoid circular dependencies.
"""

import json
import logging
import subprocess
import time
from typing import TYPE_CHECKING, Any

import httpx

from .client import KardbrdClient
from .tools import TOOLS, ToolExecutor

# Rate limit retry configuration
RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BASE_DELAY = 60  # Start with 60 seconds for per-minute rate limits
RATE_LIMIT_MAX_DELAY = 300  # Cap at 5 minutes

if TYPE_CHECKING:
    from .triggers import TriggerContext

logger = logging.getLogger(__name__)


def run_trigger(
    context: "TriggerContext",
    system_prompt: str,
    messages: list[dict[str, Any]],
    web_search_enabled: bool = False,
) -> None:
    """Entry point for running a trigger - creates resources, runs conversation.

    Args:
        context: TriggerContext with DATA only (subscription contains api_url, bot_token, etc.)
        system_prompt: The system prompt for the AI
        messages: Initial messages for the conversation
        web_search_enabled: Enable web search (Anthropic only)
    """
    if not context.subscription:
        logger.error("No subscription in context - cannot run trigger")
        return

    subscription = context.subscription

    # Create fresh instances from context DATA
    client = KardbrdClient(base_url=subscription.api_url, token=subscription.bot_token)
    tool_executor = ToolExecutor(client)

    try:
        if subscription.ai_provider == "anthropic_api":
            run_anthropic_conversation(context, system_prompt, messages, client, tool_executor, web_search_enabled)
        elif subscription.ai_provider == "gemini_api":
            run_gemini_conversation(context, system_prompt, messages, client, tool_executor)
        elif subscription.ai_provider == "gemini_cli":
            run_gemini_cli_conversation(context, system_prompt, messages, subscription)
        elif subscription.ai_provider == "mistral_vibe":
            run_mistral_vibe_conversation(context, system_prompt, messages, subscription)
        else:
            logger.error(f"Unknown AI provider: {subscription.ai_provider}")
    except Exception as e:
        logger.error(f"Error in conversation: {e}")
        try:
            client.add_comment(context.card_id, f"⚠️ **Agent Error**\n\n{e}")
        except Exception:
            logger.error("Failed to post error comment")
    finally:
        client.close()


# ============================================================================
# Anthropic API
# ============================================================================


def build_anthropic_tools(web_search_enabled: bool = False) -> list[dict[str, Any]]:
    """Build tools array in Anthropic format."""
    tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["input_schema"],
        }
        for t in TOOLS
    ]

    if web_search_enabled:
        # Web search is a special Anthropic-provided tool
        # Anthropic executes it server-side, we just include it in the schema
        tools.append(
            {
                "type": "web_search_20250305",
                "name": "web_search",
            }
        )

    return tools


def build_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert simple message format to Anthropic messages format."""
    return [
        {"role": "user", "content": msg["content"]}
        if msg["role"] == "user"
        else {"role": "assistant", "content": msg["content"]}
        for msg in messages
    ]


def _process_anthropic_tool_calls(
    tool_use_blocks: list,
    anthropic_messages: list,
    tool_executor: ToolExecutor,
    client: KardbrdClient,
    card_id: str,
    user_name: str | None = None,
) -> None:
    """Execute tool calls and update message history."""
    for tool_block in tool_use_blocks:
        tool_name = tool_block.name
        tool_input = tool_block.input or {}
        logger.info(f"Executing tool {tool_name} with args {tool_input}")

        try:
            result = tool_executor.execute(tool_name, tool_input)
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": str(result),
                        }
                    ],
                }
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool {tool_name} failed: {e}")

            # Post error comment
            _post_error_comment(
                client=client,
                card_id=card_id,
                error_message=error_msg,
                tool_name=tool_name,
                tool_input=tool_input,
                user_name=user_name,
            )

            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": f"Error: {error_msg}",
                            "is_error": True,
                        }
                    ],
                }
            )


def _call_anthropic_with_retry(
    anthropic_client,
    model: str,
    max_tokens: int,
    system: str,
    messages: list,
    tools: list,
) -> Any:
    """Call Anthropic API with exponential backoff retry for rate limits.

    Returns the response or raises the exception after retries exhausted.
    """
    import anthropic

    last_exception = None

    for attempt in range(RATE_LIMIT_MAX_RETRIES):
        try:
            return anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=tools,
            )
        except anthropic.RateLimitError as e:
            last_exception = e
            if attempt < RATE_LIMIT_MAX_RETRIES - 1:
                # Exponential backoff: 60s, 120s, 240s, 300s (capped)
                delay = min(RATE_LIMIT_BASE_DELAY * (2**attempt), RATE_LIMIT_MAX_DELAY)
                logger.warning(
                    f"Rate limited (attempt {attempt + 1}/{RATE_LIMIT_MAX_RETRIES}), waiting {delay}s before retry..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Rate limit exceeded after {RATE_LIMIT_MAX_RETRIES} retries")
                raise

    # Should not reach here, but just in case
    raise last_exception


def run_anthropic_conversation(
    context: "TriggerContext",
    system_prompt: str,
    messages: list[dict[str, Any]],
    client: KardbrdClient,
    tool_executor: ToolExecutor,
    web_search_enabled: bool = False,
) -> None:
    """Run Anthropic API conversation. No knowledge of BaseAgent."""
    import anthropic

    subscription = context.subscription
    if not subscription or not subscription.api_key:
        logger.error("No API key in subscription - cannot run Anthropic conversation")
        return

    logger.info(f"Using Anthropic API with model: {subscription.ai_model}")
    anthropic_tools = build_anthropic_tools(web_search_enabled)
    anthropic_messages = build_anthropic_messages(messages)
    anthropic_client = anthropic.Anthropic(api_key=subscription.api_key)

    for _iteration in range(10):
        try:
            response = _call_anthropic_with_retry(
                anthropic_client,
                model=subscription.ai_model,
                max_tokens=4096,
                system=system_prompt,
                messages=anthropic_messages,
                tools=anthropic_tools,
            )

            if not response.content:
                logger.error("Empty response from Anthropic API")
                break

            anthropic_messages.append({"role": "assistant", "content": response.content})

            # Filter tool_use blocks (web_search_tool_result is handled by Anthropic)
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if not tool_use_blocks:
                text_blocks = [b.text for b in response.content if b.type == "text"]
                if text_blocks:
                    final_text = "\n".join(text_blocks)
                    logger.info(f"Agent response: {final_text}")
                    messages.append({"role": "assistant", "content": final_text})
                break

            _process_anthropic_tool_calls(
                tool_use_blocks,
                anthropic_messages,
                tool_executor,
                client,
                context.card_id,
                context.user_name,
            )

        except Exception as e:
            logger.error(f"Error in Anthropic API conversation: {e}")
            raise  # Re-raise so run_trigger can post error comment


# ============================================================================
# Gemini API
# ============================================================================


def _process_gemini_tool_calls(
    tool_calls: list,
    tool_executor: ToolExecutor,
    client: KardbrdClient,
    card_id: str,
    user_name: str | None = None,
) -> list[dict]:
    """Execute tool calls and return Gemini-formatted responses."""
    tool_responses = []
    for tc in tool_calls:
        name = tc["name"]
        args = tc.get("args", {})
        logger.info(f"Executing tool {name} with args {args}")

        try:
            result = tool_executor.execute(name, args)
            tool_responses.append(
                {
                    "functionResponse": {
                        "name": name,
                        "response": {"result": result},
                    }
                }
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool {name} failed: {e}")

            # Post error comment
            _post_error_comment(
                client=client,
                card_id=card_id,
                error_message=error_msg,
                tool_name=name,
                tool_input=args,
                user_name=user_name,
            )

            tool_responses.append(
                {
                    "functionResponse": {
                        "name": name,
                        "response": {"error": error_msg},
                    }
                }
            )
    return tool_responses


def run_gemini_conversation(
    context: "TriggerContext",
    system_prompt: str,
    messages: list[dict[str, Any]],
    client: KardbrdClient,
    tool_executor: ToolExecutor,
) -> None:
    """Run a conversation with the Gemini API, handling tool calls."""
    subscription = context.subscription
    if not subscription or not subscription.api_key:
        logger.error("No API key in subscription - cannot run Gemini conversation")
        return

    logger.info(f"Using Gemini API with model: {subscription.ai_model}")

    # Gemini API format for tools
    gemini_tools = [
        {
            "function_declarations": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                }
                for t in TOOLS
            ]
        }
    ]

    gemini_history = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})

    payload = {
        "contents": gemini_history,
        "tools": gemini_tools,
        "system_instruction": {"parts": [{"text": system_prompt}]},
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{subscription.ai_model}:generateContent?key={subscription.api_key}"

    max_iterations = 10
    for _iteration in range(max_iterations):
        try:
            with httpx.Client() as http_client:
                response = http_client.post(url, json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()

                # Extract candidate
                candidates = data.get("candidates", [])
                if not candidates:
                    logger.error(f"No candidates in Gemini response: {data}")
                    break

                candidate = candidates[0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])

                # Add model response to history
                gemini_history.append(content)

                # Check for tool calls
                tool_calls = [p.get("functionCall") for p in parts if p.get("functionCall")]

                if not tool_calls:
                    # Final response
                    text_parts = [p.get("text") for p in parts if p.get("text")]
                    if text_parts:
                        final_text = "\n".join(text_parts)
                        logger.info(f"Agent response: {final_text}")
                        messages.append({"role": "assistant", "content": final_text})
                    break

                # Execute tool calls
                tool_responses = _process_gemini_tool_calls(
                    tool_calls,
                    tool_executor,
                    client,
                    context.card_id,
                    context.user_name,
                )

                # Add tool responses to history as 'user' role
                gemini_history.append({"role": "user", "parts": tool_responses})

                # Update payload for next iteration
                payload["contents"] = gemini_history

        except Exception as e:
            logger.error(f"Error in Gemini API conversation: {e}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Response body: {e.response.text}")
            break


# ============================================================================
# Gemini CLI
# ============================================================================


def run_gemini_cli_conversation(
    context: "TriggerContext",
    system_prompt: str,
    messages: list[dict[str, Any]],
    subscription: Any,  # BoardSubscription but we can't import it
) -> None:
    """Run a conversation with the gemini CLI in YOLO mode with Kardbrd MCP."""
    if not subscription:
        logger.error("No subscription - cannot run Gemini CLI conversation")
        return

    mcp_name = f"kardbrd-{subscription.board_id[:8]}"

    reg_cmd = [
        "gemini",
        "mcp",
        "add",
        mcp_name,
        "python3",
        "-m",
        "kardbrd_client.mcp_server",
        "--api-url",
        subscription.api_url,
        "--token",
        subscription.bot_token,
        "--trust",
        "--description",
        f"Kardbrd Board Tools for board {subscription.board_id}",
    ]

    try:
        subprocess.run(reg_cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to register Kardbrd MCP: {e.stderr.decode()}")

    user_prompt = "\n\n".join(
        [
            msg["content"] if isinstance(msg["content"], str) else json.dumps(msg["content"])
            for msg in messages
            if msg["role"] == "user"
        ]
    )

    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    cmd = [
        "gemini",
        full_prompt,
        "--yolo",
        "--approval-mode",
        "yolo",
        "--output-format",
        "text",
    ]

    logger.info(f"Running Gemini CLI for board {subscription.board_id}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            logger.error(f"Gemini CLI failed: {result.stderr}")
            messages.append({"role": "assistant", "content": f"Error: {result.stderr}"})
            return

        content = result.stdout.strip()
        logger.info(f"Agent response: {content}")
        messages.append({"role": "assistant", "content": content})

    except subprocess.TimeoutExpired:
        logger.error("Gemini CLI execution timed out")
        messages.append({"role": "assistant", "content": "Error: Execution timed out"})
    except Exception as e:
        logger.error(f"Unexpected error during Gemini CLI execution: {e}")
        messages.append({"role": "assistant", "content": f"Error: {str(e)}"})


# ============================================================================
# Mistral Vibe CLI
# ============================================================================


def run_mistral_vibe_conversation(
    context: "TriggerContext",
    system_prompt: str,
    messages: list[dict[str, Any]],
    subscription: Any,  # BoardSubscription but we can't import it
) -> None:
    """Run a conversation with Mistral Vibe CLI, handling tool calls."""
    import os
    import tempfile

    if not subscription or not subscription.api_key:
        logger.error("No API key in subscription - cannot run Mistral Vibe conversation")
        return

    logger.info(f"Using Mistral Vibe CLI with model: {subscription.ai_model}")

    tools_data = {
        "tools": TOOLS,
        "system_prompt": system_prompt,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(tools_data, f, indent=2)
        tools_file = f.name

    try:
        max_iterations = 10

        for iteration in range(max_iterations):
            logger.debug(f"Conversation iteration {iteration + 1}/{max_iterations}")

            user_prompt = "\n\n".join(
                [
                    msg["content"] if isinstance(msg["content"], str) else json.dumps(msg["content"])
                    for msg in messages
                    if msg["role"] == "user"
                ]
            )

            env = os.environ.copy()
            env["MISTRAL_API_KEY"] = subscription.api_key

            cmd = [
                "uv",
                "tool",
                "run",
                "--from",
                "mistral-vibe",
                "vibe",
                "-p",
                user_prompt,
                "--output",
                "json",
                "--auto-approve",
            ]

            logger.debug(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )

            if result.returncode != 0:
                logger.error(f"Vibe CLI error: {result.stderr}")
                raise RuntimeError(f"Vibe CLI failed: {result.stderr}")

            try:
                response_data = json.loads(result.stdout)
                logger.debug(f"Vibe response: {json.dumps(response_data, indent=2)}")

                if isinstance(response_data, list):
                    for msg in response_data:
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            logger.info(f"Agent response: {content}")
                            messages.append({"role": "assistant", "content": content})
                elif isinstance(response_data, dict):
                    content = response_data.get("content", "") or response_data.get("text", "")
                    logger.info(f"Agent response: {content}")
                    messages.append({"role": "assistant", "content": content})

                break

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse vibe output as JSON: {e}")
                logger.error(f"Output was: {result.stdout}")
                logger.info(f"Agent response: {result.stdout}")
                break

    finally:
        if os.path.exists(tools_file):
            os.unlink(tools_file)


# ============================================================================
# Error Handling
# ============================================================================


def _post_error_comment(
    client: KardbrdClient,
    card_id: str,
    error_message: str,
    tool_name: str | None = None,
    tool_input: dict | None = None,
    user_name: str | None = None,
) -> None:
    """Post error comment to card for user visibility.

    Format: User-friendly message + collapsible technical details.

    Args:
        client: KardbrdClient to use for posting
        card_id: The card ID to post the comment to
        error_message: The error message to display
        tool_name: Optional tool name that failed
        tool_input: Optional tool input parameters
        user_name: Optional user name to tag
    """
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

        client.add_comment(card_id, comment)
        logger.info(f"Posted error comment to card {card_id}")
    except Exception as e:
        logger.error(f"Failed to post error comment: {e}")
