"""Hermes global provider throttle and weighted workload capacity plugin."""

try:
    from .throttle import GlobalThrottle
except (ImportError, ModuleNotFoundError):
    from throttle import GlobalThrottle

_controller = None


def register(ctx):
    global _controller
    _controller = GlobalThrottle(ctx)

    # Observe the request before execution so we can use Hermes' own token estimate,
    # then reconcile from real usage after the provider returns.
    ctx.register_hook("pre_api_request", _controller.on_pre_api_request)
    ctx.register_hook("post_api_request", _controller.on_post_api_request)
    ctx.register_hook("api_request_error", _controller.on_api_request_error)

    # This is the actual throttle point. Hermes' execution middleware wraps the
    # real provider call and preserves normal retry/streaming behavior.
    ctx.register_middleware("llm_execution", _controller.wrap_llm_execution)
    ctx.register_middleware("tool_execution", _controller.wrap_tool_execution)

    # Works in the CLI and messaging gateway, including Discord.
    ctx.register_command(
        "throttle",
        _controller.handle_command,
        description="View or change the global API throttle.",
        args_hint="[status|on|off|rpm N|tpm N|rpd N|hours N|max-wait N|overrides|set <scope> ...|remove <scope>|reset|reset-learning <provider> <model>]",
    )
