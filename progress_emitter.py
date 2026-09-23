"""Live progress notification emitter for Hermes Global Throttle."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ProgressEmitter:
    """Emits live progress updates into active Hermes gateway turns or CLI."""

    def __init__(self, ctx: Any = None):
        self._ctx = ctx

    def emit_debug(self, message: str, *args: Any) -> None:
        """Log routine progress at DEBUG level without forwarding to Discord/CLI."""
        logger.debug("[throttle] " + message, *args)

    def emit(
        self,
        message: str,
        emoji: str = "⏳",
        tool_name: str = "throttle",
        session_id: str | None = None,
    ) -> bool:
        """Emit a tool-style progress message to the active turn progress bubble or CLI."""
        formatted = f"{emoji} {message}".strip()
        logger.info("[throttle] %s", formatted)

        delivered = False

        # 1. Try resolving via get_active_subagent_parent if in subagent context
        try:
            from agent.subagent_lifecycle import get_active_subagent_parent
            parent = get_active_subagent_parent()
            if parent is not None:
                cb = getattr(parent, "tool_progress_callback", None)
                if callable(cb):
                    self._deliver_to_callback(cb, formatted, tool_name, message)
                    delivered = True
        except Exception:
            pass

        if delivered:
            return True

        # 2. Try resolving via PluginManager gateway or cli_ref
        try:
            from hermes_cli.plugins import get_plugin_manager
            pm = get_plugin_manager()
            if pm is not None:
                # Gateway runner mode
                injector = getattr(pm, "_gateway_message_injector", None)
                if injector and len(injector) >= 1:
                    runner = injector[0]
                    delivered = self._deliver_via_gateway_runner(runner, formatted, tool_name, message, session_id)

                # CLI mode
                if not delivered:
                    cli = getattr(pm, "_cli_ref", None)
                    if cli is not None:
                        agent = getattr(cli, "agent", None)
                        if agent is not None:
                            cb = getattr(agent, "tool_progress_callback", None)
                            if callable(cb):
                                self._deliver_to_callback(cb, formatted, tool_name, message)
                                delivered = True
                        if not delivered and hasattr(cli, "_safe_print"):
                            cli._safe_print(f"  {formatted}")
                            delivered = True
        except Exception:
            pass

        return delivered

    def _deliver_via_gateway_runner(
        self,
        runner: Any,
        formatted: str,
        tool_name: str,
        message: str,
        session_id: str | None = None,
    ) -> bool:
        """Find the active turn in gateway runner and push to progress_queue."""
        sessions = getattr(runner, "_sessions", {})
        if not isinstance(sessions, dict):
            return False

        candidates = []
        for key, state in sessions.items():
            turn = getattr(state, "turn", None)
            if turn and getattr(turn, "agent", None) is not None:
                agent = turn.agent
                if agent not in (None, "__pending__"):
                    agent_sess_id = getattr(agent, "session_id", None)
                    if session_id is None or agent_sess_id == session_id:
                        candidates.append(agent)

        for agent in candidates:
            cb = getattr(agent, "tool_progress_callback", None)
            if callable(cb):
                turn_runner = getattr(cb, "__self__", None)
                turn_ctx = getattr(turn_runner, "_ctx", None) if turn_runner else None
                progress_q = getattr(turn_ctx, "progress_queue", None) if turn_ctx else None
                if progress_q is not None and hasattr(progress_q, "put"):
                    progress_q.put(formatted)
                    return True
                else:
                    self._deliver_to_callback(cb, formatted, tool_name, message)
                    return True
        return False

    def _deliver_to_callback(
        self,
        cb: Callable[..., Any],
        formatted: str,
        tool_name: str,
        message: str,
    ) -> None:
        """Call agent.tool_progress_callback safely."""
        turn_runner = getattr(cb, "__self__", None)
        turn_ctx = getattr(turn_runner, "_ctx", None) if turn_runner else None
        progress_q = getattr(turn_ctx, "progress_queue", None) if turn_ctx else None
        if progress_q is not None and hasattr(progress_q, "put"):
            progress_q.put(formatted)
            return

        try:
            cb("tool.started", tool_name, message)
        except Exception:
            try:
                cb("_thinking", formatted)
            except Exception:
                pass
