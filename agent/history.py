"""
agent/history.py
-----------------
Manages the conversation message history for the agentic loop.

The history is a flat list of LangChain BaseMessage objects:
  [SystemMessage, HumanMessage, AIMessage, ToolMessage, AIMessage, ...]

Each call to AgentLoop.run() adds to this history, enabling multi-turn
conversations where the model remembers prior context.

Usage
-----
    from agent.history import MessageHistory

    history = MessageHistory()
    history.add_human("Refactor the login function")
    history.add_ai(ai_message)
    history.add_tool_result(tool_call_id="abc", name="read_file", content="...")
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# Maximum number of messages to keep in history before trimming old turns.
# Each turn adds ~3-5 messages (human + ai + tool results). 80 ≈ 16-20 turns.
_MAX_MESSAGES = 80


class MessageHistory:
    """
    Flat list of BaseMessage objects representing the full conversation.

    The system prompt is stored separately and prepended on every call to
    get_messages(), so it is always the first message sent to the model
    without being duplicated in the stored history.
    """

    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        self._messages: list[BaseMessage] = []

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_human(self, text: str) -> None:
        """Append a human (user) message."""
        self._messages.append(HumanMessage(content=text))

    def add_ai(self, message: AIMessage) -> None:
        """Append an AI response (may include tool_calls)."""
        self._messages.append(message)

    def add_tool_result(
        self,
        tool_call_id: str,
        name: str,
        content: str,
    ) -> None:
        """Append a ToolMessage (result of a tool call)."""
        self._messages.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                name=name,
            )
        )

    def add_system(self, text: str) -> None:
        """Override the system prompt (rarely needed after init)."""
        self._system_prompt = text

    def append_to_system_prompt(self, text: str) -> None:
        """
        Append extra context to the system prompt.

        Used by the workspace orientation step to inject the directory listing
        without adding fake conversation turns, which confuse Groq's
        function-call JSON generation.
        """
        self._system_prompt = self._system_prompt + "\n\n" + text

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_messages(self) -> list[BaseMessage]:
        """
        Return [SystemMessage] + all stored messages.

        Automatically trims old turns if the history exceeds _MAX_MESSAGES
        to prevent unbounded context growth.
        """
        stored = self._trim(self._messages)
        return [SystemMessage(content=self._system_prompt)] + stored

    def clear(self) -> None:
        """Reset the conversation (keep the system prompt)."""
        self._messages = []

    def __len__(self) -> int:
        return len(self._messages)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trim(messages: list[BaseMessage]) -> list[BaseMessage]:
        """
        If the history is too long, drop the oldest turns while preserving
        structural validity (never cut mid-turn between an AI tool_call
        message and its ToolMessage responses).
        """
        if len(messages) <= _MAX_MESSAGES:
            return messages

        # Drop from the front, keeping the last _MAX_MESSAGES messages.
        trimmed = messages[-_MAX_MESSAGES:]

        # Ensure we don't start on an orphaned ToolMessage (which requires
        # the preceding AIMessage with tool_calls to be present).
        while trimmed and isinstance(trimmed[0], ToolMessage):
            trimmed = trimmed[1:]

        return trimmed
