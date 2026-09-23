import sys
import threading
import time
import unittest
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

# Ensure plugin root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capacity_pool import WeightedCapacityPool
from progress_emitter import ProgressEmitter
from throttle import GlobalThrottle


class DummyPluginContext:
    def __init__(self, config=None, state=None):
        self.config = config if config is not None else {}
        self.state = state if state is not None else {}
        self.hooks = {}
        self.middlewares = {}
        self.commands = {}

    def register_hook(self, hook_name, callback):
        self.hooks[hook_name] = callback

    def register_middleware(self, name, handler):
        self.middlewares[name] = handler

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = {
            "handler": handler,
            "description": description,
            "args_hint": args_hint,
        }

    def set_config(self, key, value):
        self.config[key] = value

    def set_state(self, key, value):
        self.state[key] = value

    def save_state(self):
        pass


class TestProgressEmitter(unittest.TestCase):
    def test_emit_safe_fallback(self):
        """When no agent or runner is active, emit() logs and returns False without crashing."""
        emitter = ProgressEmitter()
        result = emitter.emit("Testing safe fallback", emoji="⏳")
        self.assertFalse(result)

    def test_emit_via_subagent_parent(self):
        """When running in subagent context, emits directly to parent callback."""
        mock_parent = MagicMock()
        mock_cb = MagicMock()
        mock_parent.tool_progress_callback = mock_cb

        emitter = ProgressEmitter()
        with patch.dict("sys.modules", {"agent.subagent_lifecycle": MagicMock(get_active_subagent_parent=lambda: mock_parent)}):
            delivered = emitter.emit("Subagent capacity wait", emoji="⏳", tool_name="throttle")
            self.assertTrue(delivered)
            mock_cb.assert_called_once_with("tool.started", "throttle", "Subagent capacity wait")

    def test_emit_via_gateway_runner_progress_queue(self):
        """When in gateway runner mode, emits into turn_ctx.progress_queue."""
        q = Queue()
        turn_ctx = MagicMock()
        turn_ctx.progress_queue = q

        class DummyRunner:
            _ctx = turn_ctx

        runner_obj = DummyRunner()

        def dummy_cb(*args, **kwargs):
            pass

        dummy_cb.__self__ = runner_obj

        mock_agent = MagicMock()
        mock_agent.session_id = "test-session-123"
        mock_agent.tool_progress_callback = dummy_cb

        mock_turn = MagicMock()
        mock_turn.agent = mock_agent

        mock_session_state = MagicMock()
        mock_session_state.turn = mock_turn

        mock_runner = MagicMock()
        mock_runner._sessions = {"sess1": mock_session_state}

        mock_pm = MagicMock()
        mock_pm._gateway_message_injector = [mock_runner]

        with patch.dict("sys.modules", {"hermes_cli.plugins": MagicMock(get_plugin_manager=lambda: mock_pm)}):
            emitter = ProgressEmitter()
            delivered = emitter.emit("Waiting for token bucket", emoji="⏳", session_id="test-session-123")
            self.assertTrue(delivered)
            self.assertFalse(q.empty())
            item = q.get_nowait()
            self.assertIn("⏳ Waiting for token bucket", item)


class TestCapacityPoolStallNotifications(unittest.TestCase):
    def test_stall_and_unblock_notifications(self):
        """Acquiring past stall_threshold triggers on_stall, and unblocking triggers on_unblock."""
        # WeightedCapacityPool with capacity 1.0 (enough for 1 unit of workload)
        pool = WeightedCapacityPool(capacity=1.0)

        # Worker 1 acquires workload type 'llm_api' (weight=1.0)
        tok1 = pool.acquire("req-1", workload_type="llm_api", session_id="sess-1")
        self.assertIsNotNone(tok1)

        stalled = threading.Event()
        unblocked = threading.Event()

        def on_stall(*args, **kwargs):
            stalled.set()

        def on_unblock(*args, **kwargs):
            unblocked.set()

        def worker2():
            # Worker 2 tries to acquire with a small stall_threshold
            tok2 = pool.acquire(
                "req-2",
                workload_type="llm_api",
                session_id="sess-2",
                tool_name="terminal",
                stall_threshold=0.05,
                on_stall=on_stall,
                on_unblock=on_unblock,
            )
            pool.release("req-2")

        t2 = threading.Thread(target=worker2)
        t2.start()

        # Wait until on_stall is fired
        self.assertTrue(stalled.wait(timeout=2.0), "on_stall was not called on timeout")

        # Now release worker 1's token so worker 2 can unblock
        pool.release("req-1")

        self.assertTrue(unblocked.wait(timeout=2.0), "on_unblock was not called on resume")
        t2.join(timeout=2.0)

    def test_quick_acquire_does_not_stall(self):
        """Acquiring quickly before stall_threshold does not trigger on_stall or on_unblock."""
        pool = WeightedCapacityPool(capacity=10.0)

        stalled = threading.Event()
        unblocked = threading.Event()

        tok = pool.acquire(
            "req-quick",
            workload_type="read",
            stall_threshold=0.5,
            on_stall=lambda *args, **kwargs: stalled.set(),
            on_unblock=lambda *args, **kwargs: unblocked.set(),
        )
        self.assertIsNotNone(tok)
        pool.release("req-quick")

        self.assertFalse(stalled.is_set())
        self.assertFalse(unblocked.is_set())


class TestThrottleProgressIntegration(unittest.TestCase):
    def setUp(self):
        self.ctx = DummyPluginContext(
            config={
                "enabled": True,
                "requests_per_minute": 60,
                "tokens_per_minute": 60000,
                "adaptive_capacity": {
                    "enabled": True,
                    "base_capacity": 2.0,
                    "min_capacity": 1.0,
                    "max_capacity": 4.0,
                },
            }
        )
        self.throttle = GlobalThrottle(self.ctx)
        self.mock_emitter = MagicMock()
        self.throttle._emitter = self.mock_emitter

    def test_resource_clamp_and_recovery_emits(self):
        """Emits clamp warning on capacity drop >= 0.5, and recovery on capacity increase >= 0.5."""
        pool = self.throttle._capacity_pool
        pool.set_capacity(2.0)

        # Mock adaptive capacity returning 1.0 on observe
        with patch.object(self.throttle._adaptive_capacity, "observe", return_value=1.0):
            with patch.object(self.throttle._resource_sampler, "maybe_sample", return_value={"memory": 90.0}):
                self.throttle.refresh_resource_capacity(force_sample=True)

        self.mock_emitter.emit.assert_called_with(
            "Throttle: High host pressure detected, capacity clamped (2.0 → 1.0)",
            emoji="⚠️",
            session_id=None,
        )

        # Small fluctuation (delta = 0.2) should not emit
        self.mock_emitter.reset_mock()
        with patch.object(self.throttle._adaptive_capacity, "observe", return_value=1.2):
            with patch.object(self.throttle._resource_sampler, "maybe_sample", return_value={"memory": 85.0}):
                self.throttle.refresh_resource_capacity(force_sample=True)
        self.mock_emitter.emit.assert_not_called()

        # Recovery back to 2.0 (delta = +0.8)
        self.mock_emitter.reset_mock()
        with patch.object(self.throttle._adaptive_capacity, "observe", return_value=2.0):
            with patch.object(self.throttle._resource_sampler, "maybe_sample", return_value={"memory": 50.0}):
                self.throttle.refresh_resource_capacity(force_sample=True)
        self.mock_emitter.emit.assert_called_with(
            "Throttle: Host resources stable, capacity recovering (1.2 → 2.0)",
            emoji="📈",
            session_id=None,
        )

    def test_429_error_and_rate_learning_emits(self):
        """on_api_request_error emits alert on 429 and updates safe limit notification."""
        kwargs = {
            "error": "RateLimitError: 429 Too Many Requests",
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "session_id": "test-sess",
        }
        self.throttle.on_api_request_error(**kwargs)

        calls = self.mock_emitter.emit.call_args_list
        self.assertTrue(any(c[1].get("emoji") == "🚨" for c in calls))
        self.assertTrue(any(c[1].get("emoji") == "🧠" for c in calls))

    def test_commands_emit_progress(self):
        """Slash commands emit progress updates when modifying throttle state."""
        self.throttle.handle_command("on")
        self.mock_emitter.emit.assert_called_with("Throttle enabled", emoji="⚙️")

        self.mock_emitter.reset_mock()
        self.throttle.handle_command("off")
        self.mock_emitter.emit.assert_called_with("Throttle disabled", emoji="⚙️")

        self.mock_emitter.reset_mock()
        self.throttle.handle_command("rpm 45")
        self.mock_emitter.emit.assert_called_with("Throttle RPM set to 45", emoji="⚙️")

        self.mock_emitter.reset_mock()
        self.throttle.handle_command("set anthropic rpm 50")
        self.mock_emitter.emit.assert_called_with("Override set for anthropic: RPM=50", emoji="⚙️")

        self.mock_emitter.reset_mock()
        self.throttle.handle_command("remove anthropic rpm")
        self.mock_emitter.emit.assert_called_with("Removed RPM override for anthropic", emoji="⚙️")

    def test_routine_pacing_does_not_emit_to_discord(self):
        """Routine pacing waits (< 10s) and dispatch completion do not emit to Discord."""
        bucket = self.throttle._global_bucket
        calls = [
            (2.5, "rpm_interval", self.throttle._settings()),
            (0.0, "ok", self.throttle._settings()),
        ]

        def fake_check(*args, **kwargs):
            return calls.pop(0) if calls else (0.0, "ok", self.throttle._settings())

        with patch.object(bucket, "_check_wait_locked", side_effect=fake_check):
            with bucket._cv:
                bucket._wait_until_dispatch_allowed_locked("req-1", 100, "anthropic", "claude")

        # Discord emitter should NOT have been called for routine 2.5s pacing or completion
        self.mock_emitter.emit.assert_not_called()

    def test_unusually_long_wait_emits_to_discord(self):
        """Unusually long waits (>= 10s) emit an extended wait alert to Discord."""
        bucket = self.throttle._global_bucket
        calls = [
            (12.0, "rpd_quota", self.throttle._settings()),
            (0.0, "ok", self.throttle._settings()),
        ]

        def fake_check(*args, **kwargs):
            return calls.pop(0) if calls else (0.0, "ok", self.throttle._settings())

        with patch.object(bucket, "_check_wait_locked", side_effect=fake_check):
            with bucket._cv:
                bucket._wait_until_dispatch_allowed_locked("req-long", 100, "anthropic", "claude")

        # Emitted extended wait notice
        self.mock_emitter.emit.assert_called_once()
        call_args = self.mock_emitter.emit.call_args
        self.assertIn("Extended wait for anthropic/claude", call_args[0][0])
        self.assertEqual(call_args[1].get("emoji"), "⏳")

    def test_long_provider_pacing_names_the_limit_not_a_stale_wait(self):
        """Known provider pacing stays distinct from FIFO/capacity contention."""
        bucket = self.throttle._global_bucket
        calls = [
            (59.0, "TPM ceiling", self.throttle._settings()),
            (0.0, "ok", self.throttle._settings()),
        ]

        def fake_check(*args, **kwargs):
            return calls.pop(0) if calls else (0.0, "ok", self.throttle._settings())

        with patch.object(bucket, "_check_wait_locked", side_effect=fake_check):
            with patch("throttle.time.monotonic", side_effect=[0.0, 30.0, 31.0, 32.0]):
                with bucket._cv:
                    bucket._wait_until_dispatch_allowed_locked(
                        "req-tpm", 100, "gemini", "flash"
                    )

        self.mock_emitter.emit.assert_called_once()
        call_args = self.mock_emitter.emit.call_args
        self.assertIn("Long TPM ceiling wait for gemini/flash", call_args[0][0])
        self.assertIn("next pacing check 59.0s", call_args[0][0])
        self.assertNotIn("Stale", call_args[0][0])
        self.assertEqual(call_args[1].get("emoji"), "⏳")

    def test_burst_429_coalescing(self):
        """Burst 429 errors from multiple in-flight requests coalesce into a single backoff alert."""
        kwargs = {
            "status_code": 429,
            "provider": "openai",
            "model": "gpt-4",
            "session_id": "sess-1",
        }
        # Fire first 429
        self.throttle.on_api_request_error(**kwargs)
        # Count backoff emits (🚨)
        calls_first = [c for c in self.mock_emitter.emit.call_args_list if c[1].get("emoji") == "🚨"]
        self.assertEqual(len(calls_first), 1)

        # Fire second and third 429 immediately
        self.mock_emitter.reset_mock()
        self.throttle.on_api_request_error(**kwargs)
        self.throttle.on_api_request_error(**kwargs)
        calls_burst = [c for c in self.mock_emitter.emit.call_args_list if c[1].get("emoji") == "🚨"]
        self.assertEqual(len(calls_burst), 0, "Burst 429s within cooldown should be suppressed")

    def test_backoff_one_time_recovery_notice(self):
        """Recovering from 429 backoff emits a one-time recovery notice with ✅."""
        self.throttle._backoff_active["openai"] = True
        self.throttle.on_backoff_recovered("openai", "gpt-4")

        self.mock_emitter.emit.assert_called_with(
            "Throttle: Rate limit backoff recovered for openai/gpt-4, normal pacing restored",
            emoji="✅",
        )

        # Subsequent call does not emit duplicate notice
        self.mock_emitter.reset_mock()
        self.throttle.on_backoff_recovered("openai", "gpt-4")
        self.mock_emitter.emit.assert_not_called()

    def test_queue_buildup_and_clear_recovery(self):
        """Queue buildup (>= 3) emits a summary notice, and clearing emits a one-time notice."""
        self.throttle._check_queue_buildup("anthropic", 4)
        self.mock_emitter.emit.assert_called_with(
            "Throttle: Queue buildup: 4 requests queued for anthropic",
            emoji="⏳",
            session_id=None,
        )

        # Repeating while active does not re-spam immediately
        self.mock_emitter.reset_mock()
        self.throttle._check_queue_buildup("anthropic", 4)
        self.mock_emitter.emit.assert_not_called()

        # Draining queue emits one-time notice
        self.throttle._check_queue_cleared("anthropic")
        self.mock_emitter.emit.assert_called_with(
            "Throttle: Queue cleared for anthropic, normal pacing resumed",
            emoji="⚡",
            session_id=None,
        )

    def test_limiter_error_emits_alert(self):
        """Internal limiter errors emit an alert with ❌."""
        with patch.object(self.throttle, "acquire_workload", side_effect=RuntimeError("pool failure")):
            with self.assertRaises(RuntimeError):
                self.throttle.wrap_tool_execution(tool_name="bash", args={}, next_call=lambda x: x)

        calls = [c for c in self.mock_emitter.emit.call_args_list if c[1].get("emoji") == "❌"]
        self.assertTrue(len(calls) >= 1)
        self.assertIn("Limiter error", calls[0][0][0])


if __name__ == "__main__":
    unittest.main()
