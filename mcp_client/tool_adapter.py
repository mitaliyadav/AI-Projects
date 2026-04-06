"""
mcp/tool_adapter.py
--------------------
Converts MCP tools (mcp.types.Tool) into LangChain BaseTool instances
so that any LangChain ChatModel can bind and call them via .bind_tools().

Design
------
- Each MCP tool's JSON Schema (inputSchema) is parsed into a Pydantic model
  so LangChain can properly serialise the tool signature for the LLM.
- _arun() is the primary execution path (async throughout Phase 6).
- _run() provides a synchronous fallback using asyncio for environments
  that call tools synchronously.
- A single MCPClient instance is shared across all adapters in a session.

Usage
-----
    from mcp.tool_adapter import build_langchain_tools

    async with MCPClient(configs) as client:
        mcp_tools = client.list_all_tools()
        lc_tools  = build_langchain_tools(mcp_tools, client)
        model_with_tools = model.bind_tools(lc_tools)
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from mcp.types import Tool as MCPTool

# Import our client type (string-annotated to avoid circular imports)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mcp_client.client import MCPClient


# ------------------------------------------------------------------
# JSON Schema → Pydantic model conversion
# ------------------------------------------------------------------

# Maps JSON Schema primitive types to Python types
_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _json_schema_to_pydantic(tool_name: str, schema: dict[str, Any]) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic BaseModel from a JSON Schema dict.

    Supports: string, integer, number, boolean, array, object.
    Optional fields (not in 'required') default to None.
    """
    properties: dict[str, Any] = schema.get("properties", {})
    required_fields: list[str] = schema.get("required", [])
    field_definitions: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        raw_type = prop_schema.get("type", "string")
        description = prop_schema.get("description", "")
        python_type = _JSON_TYPE_MAP.get(raw_type, str)

        if prop_name in required_fields:
            field_definitions[prop_name] = (
                python_type,
                Field(description=description),
            )
        else:
            field_definitions[prop_name] = (
                Optional[python_type],
                Field(default=None, description=description),
            )

    # If the tool has no defined properties, accept a free-form dict
    if not field_definitions:
        field_definitions["args"] = (
            Optional[dict],
            Field(default=None, description="Tool arguments"),
        )

    model_name = f"{tool_name.replace('-', '_').title()}Input"
    return create_model(model_name, **field_definitions)


# ------------------------------------------------------------------
# LangChain tool wrapper
# ------------------------------------------------------------------

class _MCPToolAdapter(BaseTool):
    """
    LangChain BaseTool that delegates execution to an MCP server via MCPClient.

    Attributes set dynamically at construction time:
        name         : MCP tool name (used by the LLM when calling the tool).
        description  : Tool description passed to the LLM.
        args_schema  : Pydantic model derived from the MCP tool's inputSchema.
        _mcp_client  : Shared MCPClient instance (not a Pydantic field).
    """

    # These are declared as class-level annotations and set per-instance
    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = BaseModel

    # Private reference to the shared MCP client — excluded from Pydantic schema
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *, mcp_client: "MCPClient", **data: Any) -> None:
        super().__init__(**data)
        # Store client as a private attribute outside Pydantic's control
        object.__setattr__(self, "_mcp_client", mcp_client)

    # ------------------------------------------------------------------
    # Async execution — primary path used by Phase 6 agentic loop
    # ------------------------------------------------------------------
    async def _arun(self, **kwargs: Any) -> str:
        """Call the MCP tool asynchronously and return the text result."""
        client: MCPClient = object.__getattribute__(self, "_mcp_client")
        # Strip None values so optional fields don't pollute the tool call
        clean_args = {k: v for k, v in kwargs.items() if v is not None}
        return await client.call_tool(self.name, clean_args)

    # ------------------------------------------------------------------
    # Sync execution — fallback for environments without a running loop
    # ------------------------------------------------------------------
    def _run(self, **kwargs: Any) -> str:
        """
        Synchronous wrapper around _arun.
        Works when called outside an async context (e.g. tests).
        If a loop is already running, raises RuntimeError with guidance.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                f"Cannot call tool '{self.name}' synchronously from inside a "
                "running asyncio event loop. Use `await tool.arun(...)` instead, "
                "or call it from the async agentic loop."
            )
        return asyncio.run(self._arun(**kwargs))


# ------------------------------------------------------------------
# Public factory function
# ------------------------------------------------------------------

def build_langchain_tools(
    mcp_tools: list[MCPTool],
    client: "MCPClient",
) -> list[BaseTool]:
    """
    Convert a list of MCP tools into LangChain BaseTool instances.

    Parameters
    ----------
    mcp_tools : Tools returned by MCPClient.list_all_tools().
    client    : The shared MCPClient that will execute tool calls.

    Returns
    -------
    List of LangChain BaseTool objects ready to be passed to model.bind_tools().
    """
    langchain_tools: list[BaseTool] = []

    for mcp_tool in mcp_tools:
        input_schema = mcp_tool.inputSchema or {}
        # inputSchema can arrive as a dict or as a Pydantic model — normalise
        if hasattr(input_schema, "model_dump"):
            input_schema = input_schema.model_dump()

        pydantic_model = _json_schema_to_pydantic(mcp_tool.name, input_schema)

        adapter = _MCPToolAdapter(
            name=mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            args_schema=pydantic_model,
            mcp_client=client,
        )
        langchain_tools.append(adapter)

    return langchain_tools
