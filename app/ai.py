import asyncio
import copy
import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import time
from typing import Any, AsyncGenerator

import orjson
from starlette.responses import StreamingResponse

from app.tools import ToolRegistry


# ─── SSE Helpers ──────────────────────────────────────────────────────────────

def sse_event(data: dict) -> str:
    return f"data: {orjson.dumps(data).decode()}\n\n"


def sse_response(generator: AsyncGenerator[str, None]) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─── Stream Data Types ───────────────────────────────────────────────────────

@dataclass
class StreamChunk:
    """Normalized chunk from any AI provider."""
    type: str  # text | thinking_start | thinking | thinking_done | tool_use_start | tool_input_delta | tool_use_end | done | error
    content: str = ""
    tool_id: str = ""
    tool_name: str = ""
    signature: str = ""


@dataclass
class ToolCallAccumulator:
    id: str = ""
    name: str = ""
    arguments: str = ""


# ─── AI Providers ─────────────────────────────────────────────────────────────

class AIProvider(ABC):
    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
        model: str,
        thinking_budget: int = 0,
        max_tokens: int = 16384,
    ) -> AsyncGenerator[StreamChunk, None]:
        ...


class ClaudeProvider(AIProvider):
    def __init__(self, api_key: str):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=api_key)

    async def stream_chat(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
        model: str,
        thinking_budget: int = 0,
        max_tokens: int = 16384,
    ) -> AsyncGenerator[StreamChunk, None]:
        is_opus_46 = "opus-4-6" in model
        is_46_model = any(x in model for x in ["opus-4-6", "sonnet-4-6"])
        supports_thinking = thinking_budget > 0 and any(
            x in model for x in ["sonnet-4", "opus-4", "haiku-4-5"]
        )

        call_params = {
            "model": model,
            "messages": messages,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": tools if tools else None,
        }

        if is_opus_46:
            call_params["thinking"] = {"type": "adaptive"}
            call_params["temperature"] = 1
            call_params["max_tokens"] = 16384
        elif supports_thinking:
            call_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            call_params["temperature"] = 1
            call_params["max_tokens"] = max(16000, thinking_budget + 4096)
        else:
            call_params["max_tokens"] = max_tokens

        current_block_index = None
        block_data: dict[int, dict] = {}

        if is_46_model and supports_thinking:
            stream_ctx = self.client.beta.messages.stream(
                betas=["interleaved-thinking-2025-05-14"],
                **call_params,
            )
        else:
            stream_ctx = self.client.messages.stream(**call_params)

        async with stream_ctx as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    current_block_index = event.index
                    cb = event.content_block
                    if cb.type == "thinking":
                        block_data[current_block_index] = {"type": "thinking"}
                        yield StreamChunk(type="thinking_start")
                    elif cb.type == "redacted_thinking":
                        block_data[current_block_index] = {
                            "type": "redacted_thinking",
                            "data": getattr(cb, "data", ""),
                        }
                    elif cb.type == "text":
                        block_data[current_block_index] = {"type": "text"}
                    elif cb.type == "tool_use":
                        block_data[current_block_index] = {
                            "type": "tool_use",
                            "id": cb.id,
                            "name": cb.name,
                            "arguments": "",
                        }
                        yield StreamChunk(
                            type="tool_use_start",
                            tool_id=cb.id,
                            tool_name=cb.name,
                        )

                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        yield StreamChunk(
                            type="thinking", content=event.delta.thinking
                        )
                    elif event.delta.type == "signature_delta":
                        if current_block_index in block_data:
                            bd = block_data[current_block_index]
                            bd["signature"] = bd.get("signature", "") + event.delta.signature
                    elif event.delta.type == "text_delta":
                        yield StreamChunk(
                            type="text", content=event.delta.text
                        )
                    elif event.delta.type == "input_json_delta":
                        pj = event.delta.partial_json
                        if current_block_index in block_data:
                            bd = block_data[current_block_index]
                            bd["arguments"] = bd.get("arguments", "") + pj
                            yield StreamChunk(
                                type="tool_input_delta",
                                content=pj,
                                tool_id=bd.get("id", ""),
                                tool_name=bd.get("name", ""),
                            )

                elif event.type == "content_block_stop":
                    if current_block_index in block_data:
                        bd = block_data[current_block_index]
                        if bd["type"] == "thinking":
                            yield StreamChunk(
                                type="thinking_done",
                                signature=bd.get("signature", ""),
                            )
                        elif bd["type"] == "tool_use":
                            yield StreamChunk(
                                type="tool_use_end",
                                tool_id=bd.get("id", ""),
                                tool_name=bd.get("name", ""),
                            )

                elif event.type == "message_stop":
                    yield StreamChunk(type="done")


class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)

    async def stream_chat(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict],
        model: str,
        thinking_budget: int = 0,
        max_tokens: int = 16384,
    ) -> AsyncGenerator[StreamChunk, None]:
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            full_messages = messages

        is_o_series = any(model.startswith(p) for p in ["o3", "o4"])
        is_gpt5 = model.startswith("gpt-5")
        is_reasoning_model = is_o_series or is_gpt5
        is_gpt54_pro = "gpt-5.4-pro" in model

        if thinking_budget <= 0:
            reas_effort = "none" if is_gpt5 else "low"
        elif thinking_budget <= 4096:
            reas_effort = "low"
        elif thinking_budget <= 10000:
            reas_effort = "medium"
        elif thinking_budget <= 16384:
            reas_effort = "high"
        else:
            reas_effort = "xhigh"

        # gpt-5.4-pro minimum reasoning effort is "medium"
        if is_gpt54_pro and reas_effort in ("none", "low"):
            reas_effort = "medium"

        call_params = {
            "model": model,
            "messages": full_messages,
            "tools": tools if tools else None,
            "tool_choice": "auto" if tools else None,
            "stream": True,
        }

        if is_reasoning_model:
            call_params["max_completion_tokens"] = max_tokens
            call_params["reasoning_effort"] = reas_effort
            if is_gpt5:
                call_params["verbosity"] = "low"
        else:
            call_params["max_tokens"] = max_tokens

        call_params = {k: v for k, v in call_params.items() if v is not None}

        stream = await self.client.chat.completions.create(**call_params)

        reasoning_started = False
        tool_calls_data: dict[int, dict] = {}

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            reasoning_text = getattr(delta, "reasoning_content", None)
            if reasoning_text:
                if not reasoning_started:
                    reasoning_started = True
                    yield StreamChunk(type="thinking_start")
                yield StreamChunk(type="thinking", content=reasoning_text)

            if delta.content:
                if reasoning_started:
                    reasoning_started = False
                    yield StreamChunk(type="thinking_done")
                yield StreamChunk(type="text", content=delta.content)

            if delta.tool_calls:
                if reasoning_started:
                    reasoning_started = False
                    yield StreamChunk(type="thinking_done")
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_data[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_data[idx]["name"] = tc.function.name
                            yield StreamChunk(
                                type="tool_use_start",
                                tool_id=tool_calls_data[idx]["id"],
                                tool_name=tc.function.name,
                            )
                        if tc.function.arguments:
                            tool_calls_data[idx]["arguments"] += tc.function.arguments
                            yield StreamChunk(
                                type="tool_input_delta",
                                content=tc.function.arguments,
                                tool_id=tool_calls_data[idx]["id"],
                                tool_name=tool_calls_data[idx]["name"],
                            )

        if reasoning_started:
            yield StreamChunk(type="thinking_done")

        yield StreamChunk(type="done")


def get_provider(model: str, settings) -> AIProvider:
    if "claude" in model.lower():
        return ClaudeProvider(settings.anthropic_api_key)
    return OpenAIProvider(settings.openai_api_key)


# ─── Stream Collector ─────────────────────────────────────────────────────────

class StreamCollector:
    """Consumes StreamChunks from an AI provider, yields SSE events,
    and accumulates the full response for history storage."""

    def __init__(self):
        self.collected_content: str = ""
        self.collected_thinking_blocks: list[dict] = []
        self.tool_calls: dict[str, ToolCallAccumulator] = {}
        self._current_thinking: str = ""
        self._current_signature: str = ""

    async def process_stream(
        self, stream: AsyncGenerator[StreamChunk, None]
    ) -> AsyncGenerator[str, None]:
        async for chunk in stream:
            match chunk.type:
                case "text":
                    self.collected_content += chunk.content
                    yield sse_event({"type": "text", "content": chunk.content})

                case "thinking_start":
                    self._current_thinking = ""
                    self._current_signature = ""
                    yield sse_event({"type": "thinking_start"})

                case "thinking":
                    self._current_thinking += chunk.content
                    yield sse_event(
                        {"type": "thinking", "content": chunk.content}
                    )

                case "thinking_done":
                    yield sse_event({"type": "thinking_done"})
                    if self._current_thinking:
                        self.collected_thinking_blocks.append(
                            {
                                "type": "thinking",
                                "thinking": self._current_thinking,
                                "signature": chunk.signature or self._current_signature,
                            }
                        )
                    self._current_thinking = ""
                    self._current_signature = ""

                case "tool_use_start":
                    self.tool_calls[chunk.tool_id] = ToolCallAccumulator(
                        id=chunk.tool_id, name=chunk.tool_name
                    )
                    yield sse_event(
                        {"type": "tool_status", "toolName": chunk.tool_name}
                    )

                case "tool_input_delta":
                    if chunk.tool_id in self.tool_calls:
                        acc = self.tool_calls[chunk.tool_id]
                        acc.arguments += chunk.content
                        yield sse_event(
                            {
                                "type": "tool_delta",
                                "toolName": acc.name,
                                "delta": chunk.content,
                            }
                        )

                case "tool_use_end":
                    pass

                case "done":
                    pass

                case "error":
                    yield sse_event({"type": "error", "error": chunk.content})

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def build_assistant_message(self) -> dict:
        """Build history-compatible assistant message (OpenAI format for storage)."""
        tool_calls_list = []
        for acc in self.tool_calls.values():
            if acc.id and acc.name:
                tool_calls_list.append(
                    {
                        "id": acc.id,
                        "type": "function",
                        "function": {
                            "name": acc.name,
                            "arguments": acc.arguments,
                        },
                    }
                )

        msg: dict = {
            "role": "assistant",
            "content": self.collected_content,
        }
        if tool_calls_list:
            msg["tool_calls"] = tool_calls_list
        if self.collected_thinking_blocks:
            msg["_thinking_blocks"] = self.collected_thinking_blocks
        return msg

    def build_stored_message(self) -> dict:
        """Build a truncated version for history storage (large execute_python code trimmed)."""
        msg = copy.deepcopy(self.build_assistant_message())
        for tc in msg.get("tool_calls", []):
            args_str = tc.get("function", {}).get("arguments", "")
            if (
                tc.get("function", {}).get("name") == "execute_python"
                and len(args_str) > 500
            ):
                try:
                    args_obj = json.loads(args_str)
                    desc = args_obj.get("description", "Python code")
                    args_obj["code"] = (
                        f"[truncated — {len(args_str)} chars] {desc}"
                    )
                    tc["function"]["arguments"] = json.dumps(args_obj)
                except Exception:
                    tc["function"]["arguments"] = '{"code":"[truncated]"}'
        return msg

    def get_final_event(self, session_id: str) -> str:
        """Return the final SSE event: either tool_calls or done."""
        if self.has_tool_calls():
            full_msg = self.build_assistant_message()
            return sse_event(
                {
                    "type": "tool_calls",
                    "toolCalls": full_msg.get("tool_calls", []),
                    "assistantMessage": full_msg,
                    "sessionId": session_id,
                }
            )
        else:
            return sse_event({"type": "done", "sessionId": session_id})


# ─── Session Store ────────────────────────────────────────────────────────────

class SessionStore:
    """Async-compatible in-memory session store with TTL eviction."""

    def __init__(self, max_sessions: int = 500, ttl_seconds: int = 3600):
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._max_sessions = max_sessions
        self._ttl = ttl_seconds

    async def get_or_create(self, session_id: str) -> dict:
        async with self._lock:
            now = time()
            if session_id not in self._sessions:
                self._evict_expired(now)
                self._sessions[session_id] = {
                    "history": [],
                    "last_access": now,
                }
            else:
                self._sessions[session_id]["last_access"] = now
            return self._sessions[session_id]

    async def get_history(self, session_id: str) -> list[dict]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session["last_access"] = time()
                return session["history"]
            return []

    async def append_message(self, session_id: str, message: dict) -> None:
        async with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["history"].append(message)
                self._sessions[session_id]["last_access"] = time()

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    def _evict_expired(self, now: float) -> None:
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s["last_access"] > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
        while len(self._sessions) >= self._max_sessions:
            oldest = min(
                self._sessions, key=lambda k: self._sessions[k]["last_access"]
            )
            del self._sessions[oldest]


# ─── History Management ──────────────────────────────────────────────────────

def estimate_msg_tokens(msg: dict) -> int:
    tokens = len(str(msg.get("content") or "")) // 4
    for tc in msg.get("tool_calls", []):
        tokens += len(tc.get("function", {}).get("arguments", "")) // 4
    return tokens


def repair_openai_history(history: list[dict]) -> list[dict]:
    """Ensure every assistant tool_call has a matching tool result."""
    repaired = []
    i = 0
    while i < len(history):
        msg = history[i]
        repaired.append(msg)

        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_ids = {tc["id"] for tc in msg["tool_calls"]}
            provided_ids = set()
            j = i + 1
            while j < len(history) and history[j].get("role") == "tool":
                provided_ids.add(history[j].get("tool_call_id"))
                j += 1
            missing = tool_ids - provided_ids
            if missing:
                for mid in missing:
                    repaired.append(
                        {
                            "role": "tool",
                            "tool_call_id": mid,
                            "content": "Tool result missing (placeholder to satisfy API pairing requirement).",
                        }
                    )
        i += 1
    return repaired


def convert_to_claude_messages(history: list[dict]) -> list[dict]:
    """Convert OpenAI-format history to Claude message format."""
    claude_msgs = []
    current_user_content = []

    for msg in history:
        role = msg["role"]

        if role in ["user", "assistant"] and current_user_content:
            claude_msgs.append({"role": "user", "content": current_user_content})
            current_user_content = []

        if role == "user":
            claude_msgs.append({"role": "user", "content": msg["content"]})

        elif role == "assistant":
            content = []
            if msg.get("_thinking_blocks"):
                for tb in msg["_thinking_blocks"]:
                    content.append(tb)
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    args_str = tc["function"]["arguments"]
                    if not args_str:
                        tool_input = {}
                    else:
                        try:
                            tool_input = orjson.loads(args_str)
                        except Exception:
                            tool_input = {}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": tool_input,
                        }
                    )
            claude_msgs.append({"role": "assistant", "content": content})

        elif role == "tool":
            current_user_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg["content"],
                }
            )

    if current_user_content:
        claude_msgs.append({"role": "user", "content": current_user_content})

    return claude_msgs


def prepare_history(
    history: list[dict],
    system_prompt: str,
    model: str,
    max_history_messages: int = 50,
    max_history_tokens: int = 120000,
) -> tuple[list[dict], bool]:
    """Prepare conversation history for an API call."""
    is_claude = "claude" in model.lower()

    history_slice = history[-max_history_messages:]

    while history_slice and history_slice[0].get("role") == "tool":
        history_slice = history_slice[1:]

    has_tool_calls = any(
        msg.get("role") == "assistant" and msg.get("tool_calls")
        for msg in history_slice
    )

    if has_tool_calls:
        i = 0
        while i < len(history_slice) - 1:
            msg = history_slice[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                assistant_tc_ids = {tc["id"] for tc in msg["tool_calls"]}
                following_tool_ids = set()
                j = i + 1
                while j < len(history_slice) and history_slice[j].get("role") == "tool":
                    following_tool_ids.add(history_slice[j].get("tool_call_id"))
                    j += 1
                missing_ids = assistant_tc_ids - following_tool_ids
                if missing_ids:
                    for mid in missing_ids:
                        history_slice.insert(
                            j,
                            {
                                "role": "tool",
                                "tool_call_id": mid,
                                "content": "Error: Tool execution result missing. Assuming success for continuation.",
                            },
                        )
            i += 1

    system_tokens = estimate_msg_tokens({"content": system_prompt})
    while len(history_slice) > 2:
        total = system_tokens + sum(estimate_msg_tokens(m) for m in history_slice)
        if total <= max_history_tokens:
            break
        removed = history_slice.pop(0)
        if (
            removed.get("role") == "user"
            and history_slice
            and history_slice[0].get("role") == "assistant"
        ):
            history_slice.pop(0)
            while history_slice and history_slice[0].get("role") == "tool":
                history_slice.pop(0)
        elif removed.get("role") == "assistant":
            while history_slice and history_slice[0].get("role") == "tool":
                history_slice.pop(0)

    while history_slice and history_slice[0].get("role") == "tool":
        history_slice = history_slice[1:]

    if is_claude:
        repaired = repair_openai_history(history_slice)
        messages = convert_to_claude_messages(repaired)
    else:
        messages = [{"role": "system", "content": system_prompt}] + history_slice

    return messages, is_claude


# ─── Prompt Builder ───────────────────────────────────────────────────────────

def _hash_state(state: dict) -> str:
    key_parts = [
        state.get("hasAtoms", False),
        state.get("atomCount", 0),
        state.get("selectedCount", 0),
        tuple(state.get("selectedIndices", [])),
        len(state.get("fragments", [])),
        state.get("hasAxis", False),
        tuple(state.get("axisAtoms", [])) if state.get("hasAxis") else (),
        state.get("frameCount", 0),
        state.get("currentFrame", 0),
        state.get("hasEnergies", False),
        state.get("hasForces", False),
        state.get("hasMaceCache", False),
        state.get("maceFrameCount", 0),
        state.get("currentFileName", ""),
        state.get("activeScreenId"),
        state.get("screenCount", 1),
        tuple(
            (s.get("id"), s.get("atomCount", 0), s.get("frameCount", 0))
            for s in state.get("screens", [])
        ),
    ]
    return hashlib.md5(str(key_parts).encode()).hexdigest()


def _model_display_name(model: str) -> str:
    model_lower = model.lower()
    claude_match = re.match(r"claude-(opus|sonnet|haiku)-(\d+(?:-\d+)?)", model_lower)
    claude_legacy = re.match(r"claude-(\d+(?:-\d+)?)-(opus|sonnet|haiku)", model_lower)
    if claude_match:
        family = claude_match.group(1).capitalize()
        version = claude_match.group(2).replace("-", ".")
        return f"Claude {family} {version}"
    elif claude_legacy:
        family = claude_legacy.group(2).capitalize()
        version = claude_legacy.group(1).replace("-", ".")
        return f"Claude {family} {version}"
    elif "gpt-5.4" in model_lower:
        return "GPT-5.4 Pro" if "pro" in model_lower else "GPT-5.4"
    elif "gpt-5.3" in model_lower:
        return "GPT-5.3"
    elif "gpt-5.2" in model_lower:
        return "GPT-5.2" + (" Pro" if "pro" in model_lower else "")
    elif "gpt-5.1" in model_lower:
        return "GPT-5.1" + (" Codex Max" if "codex" in model_lower else "")
    elif "gpt-5" in model_lower:
        if "mini" in model_lower:
            return "GPT-5 Mini"
        elif "nano" in model_lower:
            return "GPT-5 Nano"
        else:
            return "GPT-5"
    elif "gpt-4.1" in model_lower:
        if "mini" in model_lower:
            return "GPT-4.1 Mini"
        elif "nano" in model_lower:
            return "GPT-4.1 Nano"
        else:
            return "GPT-4.1"
    elif model_lower.startswith("o4") or model_lower.startswith("o3"):
        return model.upper()
    return model


class PromptBuilder:
    """Builds and caches system prompts based on frontend state."""

    def __init__(self, max_cache: int = 1000):
        self._cache: dict[str, str] = {}
        self._max_cache = max_cache

    def build(self, state: dict, model: str = "") -> str:
        cache_key = _hash_state(state) + ":" + model
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = self._build_prompt(state, model)
        self._cache[cache_key] = prompt
        if len(self._cache) > self._max_cache:
            self._cache.pop(next(iter(self._cache)))
        return prompt

    def _build_prompt(self, state: dict, model: str) -> str:
        model_display = _model_display_name(model)

        axis_str = "None"
        if state.get("hasAxis") and len(state.get("axisAtoms", [])) == 2:
            a = state["axisAtoms"]
            axis_str = f"atoms {a[0]}-{a[1]}"

        selected_str = str(state.get("selectedCount", 0))
        if state.get("selectedCount", 0) > 0:
            selected_str += " " + str(state.get("selectedIndices", []))

        cached_str = "N"
        if state.get("hasMaceCache"):
            cached_str = f"Y({state.get('maceFrameCount', 0)})"

        folder_str = "None"
        if state.get("hasFolder"):
            folder_str = f"{len(state.get('folderFiles', []))} files"

        atom_count = state.get("atomCount", 0) if state.get("hasAtoms") else 0

        # Build screens summary for multi-screen awareness
        screens_info = state.get("screens", [])
        screens_line = ""
        if len(screens_info) > 1:
            parts = []
            for scr in screens_info:
                marker = "*" if scr.get("active") else ""
                file_part = f", file={scr.get('sourceFile')}" if scr.get("sourceFile") else ""
                parts.append(
                    f"{marker}{scr.get('title', 'Screen')}(id={scr['id']}, atoms={scr.get('atomCount', 0)}, frames={scr.get('frameCount', 0)}{file_part})"
                )
            screens_line = f"\nSCREENS: {', '.join(parts)}  (* = active)"

        return f"""ChopChopMol AI — molecular visualization and computation assistant. Powered by {model_display}.

STATE: Atoms={atom_count}, Selected={selected_str}, Axis={axis_str}, Frames={state.get('frameCount', 0)}, CachedEnergies={cached_str}, File={state.get('currentFileName') or 'None'}, Folder={folder_str}, Screen={state.get('activeScreenTitle') or 'Screen 1'}({state.get('activeScreenId', 1)}), Screens={state.get('screenCount', 1)}{screens_line}

TOOL LAYERS (compose bottom-up):
L1 QUERY: get_molecule_info, get_atom_info, get_bonded_atoms, measure_distance, measure_angle, measure_dihedral, get_frame_info, get_cached_energies, web_search, read_file, list_folder_files (read-only, no side effects)
L2 SELECT: select_atoms, select_atoms_by_element, select_all_atoms, select_connected, clear_selection (set context for L3)
L3 EDIT: add_atom, remove_atoms, change_atom_element, set_bond_distance, set_angle, set_dihedral_angle, transform_atoms, split_molecule (modify molecule, most require selection)
L4 GENERATE: rotational_scan, translation_scan, angle_scan, calculate_energy, calculate_all_energies, optimize_geometry, run_md, load_molecule, load_xyz (create frames/data)
L5 OUTPUT: create_chart, save_file, save_image, create_file, edit_file, get_xyz, execute_python (present/analyze results)
L6 VIEW: toggle_labels, toggle_force_arrows, toggle_charge_visualization, set_style, toggle_frame_info, camera, undo, redo (non-destructive)
L6.1 TRAJECTORY: go_to_frame, play_trajectory, pause_trajectory, step_forward, step_back, set_playback_speed, set_playback_mode (trajectory playback control)
L7 SCREEN: create_screen, switch_screen, duplicate_screen, delete_screen, rename_screen, get_screen_list, get_screen_info, set_screen_notes, get_screen_notes, load_file_to_screen (presentation screens)

EXECUTE_PYTHON — auto-injected variables (no need to call other tools first):
- atoms: list of {{element, x, y, z}} (current frame, Angstrom)
- positions: numpy (n_frames, n_atoms, 3) — all trajectory coordinates. Use this for vectorized analysis.
- energies: numpy 1D float array of potential energies (eV), one per frame. Plain numbers, NOT dicts.
- frames: list of {{index, atoms:[{{element,x,y,z,fx?,fy?,fz?}}]}} — only needed for element labels.
- steps, temperatures, kinetic_energies, total_energies: numpy arrays (only after MD).
- Libraries: numpy (np), matplotlib (plt), math, scipy (import yourself: from scipy.spatial.distance import pdist, cdist, etc.)
- Write EFFICIENT code: use np.linalg.norm with broadcasting, scipy.spatial.distance.pdist/cdist for pairwise distances — NEVER use Python triple-nested loops over frames×atoms×atoms.
- Do NOT call get_molecule_info or get_cached_energies before execute_python — the data is already injected.
- Write the code correctly the first time. Check the variable docs above — energies is a numpy array, not a list of dicts.

RULES:
1. Atom indices: 0-based.
2. ALWAYS ask user for MACE model (mace-mp-0a, mace-mp-0b3, mace-mpa-0) before energy/optimization/MD unless already specified.
3. Tool results include nextSteps hints — follow them for multi-step workflows.
4. If CachedEnergies=Y, use get_cached_energies instead of recalculating.
5. Brief responses (1-2 sentences). Execute tools immediately. Minimize tool calls — do as much as possible in a single execute_python call.
6. Measurement tools accept atom indices directly — no need to select first.
7. For unknown chemistry facts, use web_search. For known facts, answer directly.
8. Use the screenId parameter to operate on non-active screens without switching the user's view. Only call switch_screen when the user wants to VIEW a different screen. Tools supporting screenId: calculate_energy, calculate_all_energies, optimize_geometry, run_md, get_molecule_info, get_atom_info.
9. To load a file onto a screen, use load_file_to_screen (supports newScreen:true to create+load in one step). Use create_screen with the file param for the same effect. Always list_folder_files first if you don't know the filename.
"""


# ─── Agent Orchestrator ──────────────────────────────────────────────────────

class AgentOrchestrator:
    """Manages a single AI conversation turn within one SSE request.

    The frontend drives the outer agent loop:
    1. Frontend POSTs user message (or tool results)
    2. This orchestrator streams one AI turn back as SSE events
    3. If tool_calls, frontend executes them and POSTs results back
    4. Repeat until AI responds with text only (done event)
    """

    def __init__(
        self,
        session_store: SessionStore,
        prompt_builder: PromptBuilder,
        tool_registry: ToolRegistry,
        settings,
    ):
        self.session_store = session_store
        self.prompt_builder = prompt_builder
        self.tool_registry = tool_registry
        self.settings = settings

    async def run_turn(
        self,
        session_id: str,
        user_message: str,
        tool_results: dict | None,
        state: dict,
        model: str,
        thinking_budget: int,
    ) -> AsyncGenerator[str, None]:
        try:
            session = await self.session_store.get_or_create(session_id)
            history = session["history"]

            if tool_results is None:
                history.append({"role": "user", "content": user_message})
            else:
                for result in tool_results["results"]:
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": result["tool_call_id"],
                            "content": result["content"],
                        }
                    )
                self._ensure_tool_pairing(history, tool_results)

            system_prompt = self.prompt_builder.build(state, model)

            is_claude = "claude" in model.lower()
            messages, _ = prepare_history(history, system_prompt, model)

            if is_claude:
                tools = self.tool_registry.all_schemas_claude()
            else:
                tools = self.tool_registry.all_schemas_openai()

            provider = get_provider(model, self.settings)
            stream = provider.stream_chat(
                messages=messages,
                system_prompt=system_prompt if is_claude else "",
                tools=tools,
                model=model,
                thinking_budget=thinking_budget,
            )

            collector = StreamCollector()
            async for event in collector.process_stream(stream):
                yield event

            stored_msg = collector.build_stored_message()
            if stored_msg.get("content") or stored_msg.get("tool_calls"):
                await self.session_store.append_message(session_id, stored_msg)

            yield collector.get_final_event(session_id)

        except Exception as e:
            yield sse_event({"type": "error", "error": str(e)})

    def _ensure_tool_pairing(self, history: list[dict], tool_results: dict) -> None:
        """Ensure the tool results are properly paired with an assistant message."""
        if not history or history[-1]["role"] != "tool":
            return

        num_tools = 0
        for i in range(len(history) - 1, -1, -1):
            if history[i]["role"] == "tool":
                num_tools += 1
            else:
                break
        first_tool_idx = len(history) - num_tools

        reconstruct = True
        if first_tool_idx > 0:
            prev_msg = history[first_tool_idx - 1]
            if prev_msg["role"] == "assistant" and prev_msg.get("tool_calls"):
                assistant_tc_ids = {tc["id"] for tc in prev_msg["tool_calls"]}
                tool_tc_ids = {
                    history[j]["tool_call_id"]
                    for j in range(first_tool_idx, len(history))
                }
                if tool_tc_ids.issubset(assistant_tc_ids):
                    reconstruct = False

        if reconstruct:
            tool_ids = [
                history[j]["tool_call_id"]
                for j in range(first_tool_idx, len(history))
            ]
            reconstructed = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": "reconstructed_tool", "arguments": "{}"},
                    }
                    for tc_id in tool_ids
                ],
            }
            history.insert(first_tool_idx, reconstructed)
