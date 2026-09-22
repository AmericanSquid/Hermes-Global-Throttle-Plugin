import copy
import inspect
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Ensure plugin root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from throttle import (
    PLUGIN_STATE_KEY,
    BayesianRateLimitLearner,
    GlobalThrottle,
    _detect_429_target,
    _extract_headers,
    classify_tool_workload,
    compute_delays,
    is_local_model_execution,
    register,
)


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


class GlobalThrottleTests(unittest.TestCase):
    def setUp(self):
        self.ctx = DummyPluginContext(
            config={
                "enabled": True,
                "requests_per_minute": 60,
                "tokens_per_minute": 60000,
                "requests_per_day": 1000,
                "make_daily_quota_last_hours": 8.0,
                "longest_wait_between_requests": 15.0,
            }
        )
        self.throttle = GlobalThrottle(self.ctx)

    def test_register_contract_and_kwargs(self):
        ctx = DummyPluginContext()
        throttle = register(ctx)
        self.assertIsInstance(throttle, GlobalThrottle)

        # Hermes requires hooks: pre_api_request, post_api_request, api_request_error
        self.assertIn("pre_api_request", ctx.hooks)
        self.assertIn("post_api_request", ctx.hooks)
        self.assertIn("api_request_error", ctx.hooks)

        # Verify Hermes doctor rule: all hook callbacks must accept **kwargs
        for hook_name, cb in ctx.hooks.items():
            sig = inspect.signature(cb)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            self.assertTrue(
                has_var_kw,
                f"Hook {hook_name} callback {cb} must accept **kwargs for Hermes compatibility",
            )

        # Hermes middleware and commands
        self.assertIn("llm_execution", ctx.middlewares)
        self.assertIn("tool_execution", ctx.middlewares)
        self.assertIn("throttle", ctx.commands)

    def test_workload_classification_and_local_model_exclusion(self):
        self.assertEqual(classify_tool_workload("delegate_task"), "subagent")
        self.assertEqual(classify_tool_workload("web_search"), "api_request")
        self.assertEqual(classify_tool_workload("terminal"), "tool_call")
        self.assertIsNone(classify_tool_workload("cronjob"))
        self.assertIsNone(classify_tool_workload("cronjob_manage"))
        self.assertTrue(is_local_model_execution("ollama", ""))
        self.assertTrue(is_local_model_execution("openai", "http://127.0.0.1:11434/v1"))
        self.assertTrue(is_local_model_execution("unknown", "", "ollama/llama3"))
        self.assertFalse(is_local_model_execution("openai", "https://api.openai.com/v1"))

    def test_tool_execution_uses_and_releases_global_pool(self):
        observed = {}

        def dummy_next(args):
            observed.update(self.throttle.capacity_pool.active)
            return {"ok": args}

        result = self.throttle.wrap_tool_execution(
            tool_name="terminal",
            args={"command": "true"},
            tool_call_id="tool-1",
            next_call=dummy_next,
        )
        self.assertEqual(result, {"ok": {"command": "true"}})
        self.assertEqual(observed["tool:tool-1"]["workload_type"], "tool_call")
        self.assertEqual(self.throttle.capacity_pool.used, 0.0)

    def test_subagent_is_gated_then_released_before_child_starts(self):
        active_during_delegate = None

        def dummy_next(args):
            nonlocal active_during_delegate
            active_during_delegate = self.throttle.capacity_pool.active
            return "started"

        result = self.throttle.wrap_tool_execution(
            tool_name="delegate_task",
            args={"task": "work"},
            tool_call_id="delegate-1",
            next_call=dummy_next,
        )
        self.assertEqual(result, "started")
        self.assertEqual(active_during_delegate, {})

    def test_cron_scheduling_bypasses_capacity_pool(self):
        self.throttle.capacity_pool.set_capacity(1)
        self.throttle.capacity_pool.acquire("occupied", "interactive")
        try:
            result = self.throttle.wrap_tool_execution(
                tool_name="cronjob",
                args={"schedule": "0 * * * *"},
                tool_call_id="cron-1",
                next_call=lambda args: "scheduled",
            )
            self.assertEqual(result, "scheduled")
        finally:
            self.throttle.capacity_pool.release("occupied")

    def test_corrupted_state_recovery(self):
        # State containing non-numeric / negative garbage
        corrupted_ctx = DummyPluginContext(
            state={
                PLUGIN_STATE_KEY: {
                    "day": {"requests": -99, "tokens": None},
                    "adaptive": {
                        "factor": "not-a-float",
                        "cooldown_until": "bad-timestamp",
                        "error_streak": -5,
                    },
                    "token_stats": {
                        "avg_total_tokens": "invalid",
                        "samples": "corrupt",
                    },
                }
            }
        )
        t = GlobalThrottle(corrupted_ctx)
        self.assertGreaterEqual(t.ewma_tokens, 100.0)
        self.assertEqual(t.request_count, 0)
        self.assertEqual(t.token_count, 0)
        self.assertAlmostEqual(t.adaptive_factor, 1.0)
        self.assertEqual(t.cooldown_until, 0.0)
        self.assertEqual(t.error_streak, 0)

    def test_settings_coercion(self):
        # Test boolean string parsing and numeric conversions
        self.ctx.config["enabled"] = "false"
        self.ctx.config["requests_per_minute"] = "120"
        self.ctx.config["tokens_per_minute"] = "100000"
        s = self.throttle._settings()
        self.assertFalse(s.enabled)
        self.assertEqual(s.rpm, 120)
        self.assertEqual(s.tpm, 100000)

        self.ctx.config["enabled"] = "0"
        s = self.throttle._settings()
        self.assertFalse(s.enabled)

        self.ctx.config["enabled"] = "True"
        s = self.throttle._settings()
        self.assertTrue(s.enabled)

    def test_pre_and_post_api_request_flow(self):
        request = {
            "messages": [
                {"role": "user", "content": "Hello, how are you doing today?"}
            ]
        }
        res = self.throttle.on_pre_api_request(request=request)
        req_id = res.get("request_id")
        self.assertIsNotNone(req_id)
        self.assertIn(req_id, self.throttle._pending_requests)

        # Simulate post API response with dict usage
        response = {
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 12,
                "total_tokens": 20,
            }
        }
        self.throttle.on_post_api_request(
            request=request,
            response=response,
            request_id=req_id,
        )
        self.assertEqual(self.throttle.token_count, 20)
        self.assertNotIn(req_id, self.throttle._pending_requests)

    def test_pending_estimate_uses_provider_model_token_history(self):
        self.throttle.on_post_api_request(
            request_id="small-sample",
            provider="provider-a",
            model="model-a",
            response={
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 90,
                    "total_tokens": 100,
                }
            },
        )
        self.throttle.on_post_api_request(
            request_id="large-sample",
            provider="provider-b",
            model="model-b",
            response={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 900,
                    "total_tokens": 1000,
                }
            },
        )

        small_with_input = self.throttle.on_pre_api_request(
            request_id="small-with-input",
            provider="provider-a",
            model="model-a",
            approx_input_tokens=10,
        )
        large_with_input = self.throttle.on_pre_api_request(
            request_id="large-with-input",
            provider="provider-b",
            model="model-b",
            approx_input_tokens=10,
        )
        small_without_input = self.throttle.on_pre_api_request(
            request_id="small-without-input",
            provider="provider-a",
            model="model-a",
        )
        large_without_input = self.throttle.on_pre_api_request(
            request_id="large-without-input",
            provider="provider-b",
            model="model-b",
        )

        self.assertEqual(small_with_input["estimated_tokens"], 100)
        self.assertEqual(large_with_input["estimated_tokens"], 910)
        self.assertEqual(small_without_input["estimated_tokens"], 100)
        self.assertEqual(large_without_input["estimated_tokens"], 1000)

    def test_pending_estimate_uses_global_average_without_scope_history(self):
        self.throttle.on_post_api_request(
            request_id="sample",
            provider="known-provider",
            model="known-model",
            response={
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 80,
                    "total_tokens": 100,
                }
            },
        )

        with_input = self.throttle.on_pre_api_request(
            request_id="unseen-with-input",
            provider="unseen-provider",
            model="unseen-model",
            approx_input_tokens=10,
        )
        without_input = self.throttle.on_pre_api_request(
            request_id="unseen-without-input",
            provider="unseen-provider",
            model="unseen-model",
        )

        self.assertEqual(with_input["estimated_tokens"], 90)
        self.assertEqual(without_input["estimated_tokens"], 100)

    def test_post_api_request_with_object_usage(self):
        req_id = "req_obj_test"
        self.throttle._pending_requests[req_id] = SimpleNamespace(
            provider="test",
            model="test-model",
            approx_input_tokens=15,
            estimated_total_tokens=50,
            created_at=time.time(),
        )
        # Object style usage (OpenAI/Anthropic SDKs)
        usage_obj = SimpleNamespace(prompt_tokens=15, completion_tokens=35)
        response = SimpleNamespace(usage=usage_obj)

        self.throttle.on_post_api_request(
            request={},
            response=response,
            request_id=req_id,
        )
        self.assertEqual(self.throttle.token_count, 50)

    def test_on_api_request_error_429_detection(self):
        # Trigger 429 via status code
        error_context = {"status_code": 429, "error": "Too Many Requests"}
        self.throttle.on_api_request_error(**error_context)
        self.assertEqual(self.throttle.error_streak, 1)
        self.assertLess(self.throttle.adaptive_factor, 1.0)
        self.assertGreater(self.throttle.cooldown_until, time.time())

        # Trigger second 429 via error message string
        error_context_msg = {"error": "Rate limit exceeded (HTTP 429)"}
        self.throttle.on_api_request_error(**error_context_msg)
        self.assertEqual(self.throttle.error_streak, 2)

    def test_wrap_llm_execution_disabled(self):
        self.ctx.config["enabled"] = False
        called = False

        def dummy_next(req):
            nonlocal called
            called = True
            return {"result": "ok"}

        result = self.throttle.wrap_llm_execution(
            {"request": {}, "next_call": dummy_next}
        )
        self.assertTrue(called)
        self.assertEqual(result, {"result": "ok"})

    def test_remote_llm_execution_uses_capacity_pool(self):
        observed = {}

        def dummy_next(req):
            observed.update(self.throttle.capacity_pool.active)
            return "ok"

        result = self.throttle.wrap_llm_execution({
            "request": {},
            "next_call": dummy_next,
            "request_id": "api-1",
            "provider": "openai",
            "model": "gpt-test",
            "base_url": "https://api.openai.com/v1",
        })
        self.assertEqual(result, "ok")
        self.assertEqual(observed["llm:api-1"]["workload_type"], "llm_api")
        self.assertEqual(self.throttle.capacity_pool.used, 0.0)

    def test_provider_dispatch_is_not_committed_until_capacity_is_available(self):
        self.ctx.config["capacity_units"] = 5
        self.throttle.acquire_workload("blocker", "llm_api")
        started = threading.Event()

        def run_request():
            self.throttle.wrap_llm_execution({
                "request": {},
                "next_call": lambda request: started.set() or "ok",
                "request_id": "joint-admission",
                "provider": "openai",
                "model": "gpt-test",
                "base_url": "https://api.openai.com/v1",
            })

        thread = threading.Thread(target=run_request)
        thread.start()
        self.assertFalse(started.wait(timeout=0.05))
        # Read the internal state directly: the joint-admission thread owns the
        # bucket lock while waiting, and public accessors correctly share it.
        self.assertEqual(self.throttle._state["day"]["requests"], 0)
        self.assertEqual(self.throttle._global_bucket._reservations, {})

        self.throttle.release_workload("blocker")
        self.assertTrue(started.wait(timeout=1.0))
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.throttle._state["day"]["requests"], 1)

    def test_local_llm_execution_bypasses_capacity_pool(self):
        observed = None

        def dummy_next(req):
            nonlocal observed
            observed = self.throttle.capacity_pool.active
            return "ok"

        result = self.throttle.wrap_llm_execution({
            "request": {},
            "next_call": dummy_next,
            "request_id": "local-1",
            "provider": "ollama",
            "model": "llama",
            "base_url": "http://localhost:11434/v1",
        })
        self.assertEqual(result, "ok")
        self.assertEqual(observed, {})

    def test_wrap_llm_execution_missing_params(self):
        # Should gracefully return None/request without crashing
        result = self.throttle.wrap_llm_execution({})
        self.assertIsNone(result)

    def test_commands_status_and_toggle(self):
        res = self.throttle.handle_command(["status"])
        self.assertIn("Global Throttle Status", res)

        res = self.throttle.handle_command(["off"])
        self.assertIn("disabled", res)
        self.assertFalse(self.ctx.config.get("enabled"))

        res = self.throttle.handle_command(["on"])
        self.assertIn("enabled", res)
        self.assertTrue(self.ctx.config.get("enabled"))

    def test_commands_set_rates(self):
        res = self.throttle.handle_command(["rpm", "45"])
        self.assertIn("RPM set to 45", res)
        self.assertEqual(self.ctx.config.get("requests_per_minute"), 45)

        res = self.throttle.handle_command(["tpm", "75000"])
        self.assertIn("TPM set to 75000", res)
        self.assertEqual(self.ctx.config.get("tokens_per_minute"), 75000)

        res = self.throttle.handle_command(["rpd", "1500"])
        self.assertIn("RPD set to 1500", res)
        self.assertEqual(self.ctx.config.get("requests_per_day"), 1500)

        res = self.throttle.handle_command(["hours", "10"])
        self.assertIn("quota-hours set to 10", res)
        self.assertEqual(self.ctx.config.get("make_daily_quota_last_hours"), 10.0)

        res = self.throttle.handle_command(["max-wait", "25"])
        self.assertIn("max-wait set to 25", res)
        self.assertEqual(self.ctx.config.get("longest_wait_between_requests"), 25.0)

        res = self.throttle.handle_command(["reset"])
        self.assertIn("Adaptive rate factor and cooldown have been reset", res)
        self.assertAlmostEqual(self.throttle.adaptive_factor, 1.0)
        self.assertEqual(self.throttle.error_streak, 0)

    def test_commands_invalid_input(self):
        res = self.throttle.handle_command(["unknown_command"])
        self.assertIn("Usage:", res)

        res = self.throttle.handle_command(["rpm", "not_a_number"])
        self.assertIn("Invalid RPM value", res)

        res = self.throttle.handle_command(["hours", "-5"])
        self.assertIn("cannot be negative", res)

    def test_tpm_blocks_when_full_and_frees_when_expired(self):
        # Configure TPM limit to 1000 with very high RPM/RPD so only TPM is the ceiling
        ctx = DummyPluginContext(
            config={
                "enabled": True,
                "requests_per_minute": 60000,
                "tokens_per_minute": 1000,
                "requests_per_day": 100000,
                "make_daily_quota_last_hours": 0.0,
                "longest_wait_between_requests": 0.0,
            }
        )
        throttle = GlobalThrottle(ctx)

        # Complete a prior request with 800 tokens at current time
        t_now = time.time()
        with throttle._cv:
            throttle._token_ledger.append((t_now, 800))

        # Next request estimates 300 tokens: 800 (used) + 300 (estimate) = 1100 > 1000 TPM
        throttle._pending["req2"] = SimpleNamespace(
            provider="test",
            model="test-model",
            approx_input_tokens=0,
            estimated_total_tokens=300,
            created_at=t_now,
        )

        executed = []
        def dispatch_worker():
            res = throttle.wrap_llm_execution(
                {"request": {"id": "req2"}, "next_call": lambda r: executed.append(r), "request_id": "req2"}
            )
            return res

        worker_thread = threading.Thread(target=dispatch_worker)
        worker_thread.start()

        # Give the worker a moment to hit the throttle
        worker_thread.join(timeout=0.1)

        # TPM should be full, so the request remains blocked
        self.assertTrue(worker_thread.is_alive())
        self.assertEqual(executed, [])
        with throttle._cv:
            self.assertEqual(throttle._last_delay_reason, "TPM ceiling")

        # Now age the old entry past 60 seconds (expired)
        with throttle._cv:
            throttle._token_ledger = [(t_now - 65.0, 800)]
            throttle._cv.notify_all()

        # Worker should now unblock as expired usage is freed
        worker_thread.join(timeout=2.0)
        self.assertFalse(worker_thread.is_alive())
        self.assertEqual(len(executed), 1)

    def test_tpm_in_flight_reservation_blocks_and_frees_on_completion_or_error(self):
        ctx = DummyPluginContext(
            config={
                "enabled": True,
                "requests_per_minute": 60000,
                "tokens_per_minute": 1000,
                "requests_per_day": 100000,
                "make_daily_quota_last_hours": 0.0,
                "longest_wait_between_requests": 0.0,
            }
        )
        throttle = GlobalThrottle(ctx)

        # Request 1 dispatches with estimate 700
        throttle._pending["req1"] = SimpleNamespace(
            provider="test",
            model="test-model",
            approx_input_tokens=0,
            estimated_total_tokens=700,
            created_at=time.time(),
        )
        throttle.wrap_llm_execution(
            {"request": {"id": "req1"}, "next_call": lambda r: r, "request_id": "req1"}
        )

        # Verify req1 is reserved in flight
        self.assertEqual(throttle.reservations.get("req1"), 700)

        # Reset last_dispatch_mono into the past so smooth pacing delay has already elapsed,
        # isolating the in-flight reservation ceiling.
        with throttle._cv:
            throttle._last_dispatch_mono = time.monotonic() - 60.0
            throttle._global_bucket._burst_credit["test::test-model"] = (
                2.0, time.monotonic()
            )

        # Request 2 arrives with estimate 400 (700 reserved + 400 estimated = 1100 > 1000 TPM)
        throttle._pending["req2"] = SimpleNamespace(
            provider="test",
            model="test-model",
            approx_input_tokens=0,
            estimated_total_tokens=400,
            created_at=time.time(),
        )

        executed = []
        def dispatch_worker():
            throttle.wrap_llm_execution(
                {"request": {"id": "req2"}, "next_call": lambda r: executed.append(r), "request_id": "req2"}
            )

        worker_thread = threading.Thread(target=dispatch_worker)
        worker_thread.start()

        worker_thread.join(timeout=0.1)
        # Blocked because in-flight reservation exceeds capacity
        self.assertTrue(worker_thread.is_alive())
        self.assertEqual(executed, [])

        # Request 1 fails/cancels, releasing reservation
        throttle.on_api_request_error(request_id="req1")
        self.assertNotIn("req1", throttle.reservations)

        # Worker 2 now proceeds
        worker_thread.join(timeout=2.0)
        self.assertFalse(worker_thread.is_alive())
        self.assertEqual(len(executed), 1)

        # Complete request 2 and check ledger replaces reservation
        throttle.on_post_api_request(
            request_id="req2",
            response={"usage": {"total_tokens": 350}},
        )
        self.assertNotIn("req2", throttle.reservations)
        self.assertEqual(len(throttle.token_ledger), 1)
        self.assertEqual(throttle.token_ledger[0][1], 350)

    def test_bayesian_learner_initialization_and_priors(self):
        ctx = DummyPluginContext(
            config={"requests_per_minute": 40, "tokens_per_minute": 80000}
        )
        t = GlobalThrottle(ctx)
        entry = t.learner.get_or_create_entry("anthropic", "claude-3-opus", default_rpm=40, default_tpm=80000)
        self.assertEqual(entry["provider"], "anthropic")
        self.assertEqual(entry["model"], "claude-3-opus")
        self.assertEqual(len(entry["rpm"]["particles"]), 30)
        self.assertEqual(len(entry["tpm"]["particles"]), 30)
        self.assertAlmostEqual(entry["rpm"]["mean"], 40.0, delta=1.0)
        self.assertAlmostEqual(entry["tpm"]["mean"], 80000.0, delta=2000.0)
        self.assertEqual(entry["confidence"], 0.0)

        eff_rpm, eff_tpm = t.learner.get_effective_limits("anthropic", "claude-3-opus", default_rpm=40, default_tpm=80000)
        self.assertEqual(eff_rpm, 40)
        self.assertEqual(eff_tpm, 80000)

    def test_bayesian_learner_success_and_429_updates(self):
        import json
        ctx = DummyPluginContext(
            config={"requests_per_minute": 30, "tokens_per_minute": 60000}
        )
        t = GlobalThrottle(ctx)
        entry = t.learner.get_or_create_entry("openai", "gpt-4o", default_rpm=30, default_tpm=60000)
        initial_rpm_mean = entry["rpm"]["mean"]
        initial_tpm_mean = entry["tpm"]["mean"]

        now = time.time()
        for _ in range(16):
            t.learner.on_success(
                provider="openai",
                model="gpt-4o",
                actual_tokens=5000,
                rolling_requests=50,
                rolling_tokens=100000,
                now=now,
                default_rpm=30,
                default_tpm=60000,
            )

        updated_entry = t.learner.get_or_create_entry("openai", "gpt-4o", default_rpm=30, default_tpm=60000)
        # Configured RPM/TPM are absolute policy ceilings: learner never exceeds them
        # Successful traffic does not push learned ceiling upward
        self.assertEqual(updated_entry["rpm"]["mean"], initial_rpm_mean)
        self.assertEqual(updated_entry["tpm"]["mean"], initial_tpm_mean)
        # Successful traffic increases confidence
        self.assertGreaterEqual(updated_entry["confidence"], 0.70)

        eff_rpm, eff_tpm = t.learner.get_effective_limits("openai", "gpt-4o", default_rpm=30, default_tpm=60000)
        self.assertEqual(eff_rpm, updated_entry["safe_rpm"])
        self.assertEqual(eff_tpm, updated_entry["safe_tpm"])
        self.assertEqual(eff_rpm, 30)

        pre_429_rpm = updated_entry["rpm"]["mean"]
        t.learner.on_429(
            provider="openai",
            model="gpt-4o",
            estimated_tokens=3000,
            rolling_requests=35,
            rolling_tokens=50000,
            now=now,
            default_rpm=30,
            default_tpm=60000,
        )
        post_429_entry = t.learner.get_or_create_entry("openai", "gpt-4o", default_rpm=30, default_tpm=60000)
        self.assertLess(post_429_entry["rpm"]["mean"], pre_429_rpm)
        self.assertLess(post_429_entry["safe_rpm"], 30)
        self.assertGreater(post_429_entry["slowdown_until"], now)

        # After 429 lowers the limit, keep lower limit in place instead of recovering
        lowered_rpm = post_429_entry["safe_rpm"]
        for _ in range(16):
            t.learner.on_success(
                provider="openai",
                model="gpt-4o",
                actual_tokens=1000,
                rolling_requests=lowered_rpm,
                rolling_tokens=10000,
                now=now,
                default_rpm=30,
                default_tpm=60000,
            )
        post_recovery_entry = t.learner.get_or_create_entry("openai", "gpt-4o", default_rpm=30, default_tpm=60000)
        self.assertEqual(post_recovery_entry["safe_rpm"], lowered_rpm)
        self.assertGreaterEqual(post_recovery_entry["confidence"], 0.70)
        eff_rpm, eff_tpm = t.learner.get_effective_limits("openai", "gpt-4o", default_rpm=30, default_tpm=60000)
        # High confidence learned lower value continues to take precedence over configured ceiling
        self.assertEqual(eff_rpm, lowered_rpm)

        with t._cv:
            t._persist_locked()
        learned_state = ctx.state.get(PLUGIN_STATE_KEY, {}).get("learned_limits", {})
        self.assertIn("openai::gpt-4o", learned_state)
        serialized = json.dumps(ctx.state)
        self.assertNotIn("_token_ledger", serialized)
        self.assertNotIn("_reservations", serialized)

    def test_commands_status_and_reset_learning(self):
        ctx = DummyPluginContext(
            config={"requests_per_minute": 30, "tokens_per_minute": 60000}
        )
        t = GlobalThrottle(ctx)
        t.learner.on_success("google", "gemini-1.5-pro", 1000, 20, 20000, time.time(), 30, 60000)
        with t._cv:
            t._persist_locked()
        status_res = t.handle_command(["status"])
        self.assertIn("Configured global limits:", status_res)
        self.assertIn("### Learned safe limits", status_res)
        self.assertIn("google::gemini-1.5-pro", status_res)
        self.assertIn("calibrating", status_res)
        self.assertIn("Diagnostics", t.handle_command(["status", "verbose"]))

        reset_res = t.handle_command(["reset-learning", "google", "gemini-1.5-pro"])
        self.assertIn("Learned limits reset for google::gemini-1.5-pro", reset_res)
        self.assertNotIn("google::gemini-1.5-pro", t.learned_limits)

        reset_res_none = t.handle_command(["reset-learning", "google", "gemini-1.5-pro"])
        self.assertIn("No learned limits found", reset_res_none)

    def test_recalibrate_command_reopens_only_the_target_model(self):
        ctx = DummyPluginContext(
            config={"requests_per_minute": 50, "tokens_per_minute": 100000}
        )
        throttle = GlobalThrottle(ctx)
        now = time.time()
        for provider, model in (("target", "model"), ("other", "model")):
            for offset in range(11):
                throttle.learner.on_success(
                    provider, model, 1000, 20, 10000, now + offset, 50, 100000
                )

        throttle.learner.on_429(
            "target", "model", 1000, 20, 10000, now + 20, 50, 100000,
            target_dimension="rpm",
        )
        target = throttle.learner.get_or_create_entry("target", "model", 50, 100000)
        other = throttle.learner.get_or_create_entry("other", "model", 50, 100000)
        target_before = {
            "safe_rpm": target["safe_rpm"],
            "observation_count": target["observation_count"],
            "success_count": target["success_count"],
            "error_429_count": target["error_429_count"],
            "particles": list(target["rpm"]["particles"]),
        }
        other_before = copy.deepcopy(other)

        result = throttle.handle_command(["recalibrate", "target::model"])

        self.assertIn("Recalibration started for target::model", result)
        self.assertFalse(target["safe_ceiling_established"])
        self.assertIsNone(target["established_safe_rpm"])
        self.assertEqual(target["confidence"], 0.0)
        self.assertEqual(target["successes_since_429"], 0)
        for key, value in target_before.items():
            self.assertEqual(target[key] if key != "particles" else target["rpm"]["particles"], value)
        self.assertEqual(other, other_before)
        self.assertIn("target::model", ctx.state[PLUGIN_STATE_KEY]["learned_limits"])

    def test_learned_limits_take_precedence_in_dispatch(self):
        ctx = DummyPluginContext(
            config={
                "enabled": True,
                "requests_per_minute": 60000,
                "tokens_per_minute": 100000,
                "requests_per_day": 100000,
                "make_daily_quota_last_hours": 0.0,
                "longest_wait_between_requests": 0.0,
            }
        )
        t = GlobalThrottle(ctx)

        # Provider learned limit is trained to be very restrictive on TPM (e.g. 1000 TPM)
        entry = t.learner.get_or_create_entry("special", "model-x", default_rpm=60000, default_tpm=1000)
        entry["rpm"]["particles"] = [60000.0] * 30
        entry["rpm"]["weights"] = [1.0 / 30] * 30
        entry["rpm"]["safe_limit"] = 60000
        entry["rpm"]["confidence"] = 0.90
        entry["tpm"]["particles"] = [1000.0] * 30
        entry["tpm"]["weights"] = [1.0 / 30] * 30
        entry["tpm"]["safe_limit"] = 1000
        entry["tpm"]["confidence"] = 0.90
        entry["safe_rpm"] = 60000
        entry["safe_tpm"] = 1000
        entry["confidence"] = 0.90

        # Global TPM is 100,000, so 800 + 400 = 1200 would normally pass.
        # But with learned TPM = 1000, it exceeds the learned TPM ceiling and blocks.
        t_now = time.time()
        with t._cv:
            t._token_ledger.append((t_now, 800))

        t._pending["req_learned"] = SimpleNamespace(
            provider="special",
            model="model-x",
            approx_input_tokens=0,
            estimated_total_tokens=400,
            created_at=t_now,
        )

        executed = []
        def worker():
            t.wrap_llm_execution(
                {"request": {"id": "req_learned"}, "next_call": lambda r: executed.append(r), "request_id": "req_learned", "provider": "special", "model": "model-x"}
            )

        worker_th = threading.Thread(target=worker)
        worker_th.start()
        worker_th.join(timeout=0.1)

        # Proves learned limits took precedence over global limits and blocked
        self.assertTrue(worker_th.is_alive())
        self.assertEqual(executed, [])

        # Age the ledger entry past 60s
        with t._cv:
            t._token_ledger = [(t_now - 65.0, 800)]
            t._cv.notify_all()

        worker_th.join(timeout=2.0)
        self.assertFalse(worker_th.is_alive())
        self.assertEqual(len(executed), 1)

    def test_weighted_percentile_non_gaussian_distribution(self):
        # Asymmetric, heavily skewed particle distribution
        # Gaussian formula mu - 1.28*sigma yields ~31.91
        # Direct weighted 10th percentile is exactly 25.0
        particles = [10.0, 20.0, 30.0, 40.0, 50.0]
        weights = [0.05, 0.04, 0.02, 0.09, 0.80]

        mean, std = BayesianRateLimitLearner._compute_stats(particles, weights)
        gaussian_approx = round(mean - 1.28 * std, 2)
        actual_10th = BayesianRateLimitLearner._weighted_percentile(particles, weights, q=0.10)

        self.assertEqual(actual_10th, 25.0)
        self.assertNotEqual(actual_10th, gaussian_approx)
        self.assertAlmostEqual(gaussian_approx, 31.91, delta=0.5)

        # Uniform weights test
        unif_particles = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        unif_weights = [0.10] * 10
        # 10th percentile corresponds to particle 0 (cum=0.10)
        self.assertEqual(BayesianRateLimitLearner._weighted_percentile(unif_particles, unif_weights, q=0.10), 10.0)

        # Single particle edge case
        self.assertEqual(BayesianRateLimitLearner._weighted_percentile([42.0], [1.0], q=0.10), 42.0)

    def test_detect_429_target_from_headers_and_errors(self):
        # 1. OpenAI remaining tokens = 0 -> TPM
        h1 = {"x-ratelimit-remaining-tokens": "0", "x-ratelimit-remaining-requests": "15"}
        self.assertEqual(_detect_429_target(h1), "tpm")

        # 2. OpenAI remaining requests = 0 -> RPM
        h2 = {"x-ratelimit-remaining-tokens": "5000", "x-ratelimit-remaining-requests": "0"}
        self.assertEqual(_detect_429_target(h2), "rpm")

        # 3. Anthropic remaining headers
        h3 = {"anthropic-ratelimit-tokens-remaining": "0", "anthropic-ratelimit-requests-remaining": "5"}
        self.assertEqual(_detect_429_target(h3), "tpm")
        h4 = {"anthropic-ratelimit-tokens-remaining": "1000", "anthropic-ratelimit-requests-remaining": "0"}
        self.assertEqual(_detect_429_target(h4), "rpm")

        # 4. Standard draft RFC headers
        h5 = {"ratelimit-remaining-tokens": "0", "ratelimit-remaining-requests": "50"}
        self.assertEqual(_detect_429_target(h5), "tpm")

        # 5. Explicit type/resource headers
        h6 = {"x-ratelimit-type": "tokens"}
        self.assertEqual(_detect_429_target(h6), "tpm")
        h7 = {"ratelimit-resource": "requests"}
        self.assertEqual(_detect_429_target(h7), "rpm")

        # 6. Reset headers
        h8 = {"x-ratelimit-reset-tokens": "30s", "x-ratelimit-reset-requests": "0s"}
        self.assertEqual(_detect_429_target(h8), "tpm")
        h9 = {"x-ratelimit-reset-tokens": "0s", "x-ratelimit-reset-requests": "10s"}
        self.assertEqual(_detect_429_target(h9), "rpm")

        # 7. Error message and reason text fallback
        self.assertEqual(_detect_429_target({}, error_str="Rate limit reached for TPM"), "tpm")
        self.assertEqual(_detect_429_target({}, error_str="Rate limit exceeded: requests per minute"), "rpm")

        # 8. Generic 429 without distinguishing info -> None
        self.assertIsNone(_detect_429_target({}, error_str="429 Too Many Requests"))
        self.assertIsNone(_detect_429_target({"content-type": "application/json"}))

    def test_429_header_directed_updates_tpm_only(self):
        ctx = DummyPluginContext(config={"requests_per_minute": 30, "tokens_per_minute": 60000})
        t = GlobalThrottle(ctx)

        entry = t.learner.get_or_create_entry("provider_a", "model_a", default_rpm=30, default_tpm=60000)
        pre_rpm_mean = entry["rpm"]["mean"]
        pre_rpm_safe = entry["safe_rpm"]
        pre_tpm_mean = entry["tpm"]["mean"]

        # Call on_api_request_error with headers pointing to token limit
        t.on_api_request_error(
            status_code=429,
            provider="provider_a",
            model="model_a",
            headers={"x-ratelimit-remaining-tokens": "0", "x-ratelimit-remaining-requests": "20"},
        )

        post_entry = t.learner.get_or_create_entry("provider_a", "model_a", default_rpm=30, default_tpm=60000)
        # TPM was penalized
        self.assertLess(post_entry["tpm"]["mean"], pre_tpm_mean)
        # RPM was NOT penalized or modified
        self.assertEqual(post_entry["rpm"]["mean"], pre_rpm_mean)
        self.assertEqual(post_entry["safe_rpm"], pre_rpm_safe)

    def test_429_header_directed_updates_rpm_only(self):
        ctx = DummyPluginContext(config={"requests_per_minute": 30, "tokens_per_minute": 60000})
        t = GlobalThrottle(ctx)

        entry = t.learner.get_or_create_entry("provider_b", "model_b", default_rpm=30, default_tpm=60000)
        pre_rpm_mean = entry["rpm"]["mean"]
        pre_tpm_mean = entry["tpm"]["mean"]
        pre_tpm_safe = entry["safe_tpm"]

        # Call on_api_request_error with headers pointing to request limit
        t.on_api_request_error(
            status_code=429,
            provider="provider_b",
            model="model_b",
            headers={"x-ratelimit-remaining-requests": "0", "x-ratelimit-remaining-tokens": "50000"},
        )

        post_entry = t.learner.get_or_create_entry("provider_b", "model_b", default_rpm=30, default_tpm=60000)
        # RPM was penalized
        self.assertLess(post_entry["rpm"]["mean"], pre_rpm_mean)
        # TPM was NOT penalized or modified
        self.assertEqual(post_entry["tpm"]["mean"], pre_tpm_mean)
        self.assertEqual(post_entry["safe_tpm"], pre_tpm_safe)

    def test_generic_429_proximity_directed_updates_closest_boundary(self):
        ctx = DummyPluginContext(config={"requests_per_minute": 30, "tokens_per_minute": 60000})
        t = GlobalThrottle(ctx)

        now = time.time()
        # Scenario A: High RPM load (28/30 = 93%), low TPM load (1000/60000 = 1.6%)
        entry_a = t.learner.get_or_create_entry("prov_a", "mod_a", default_rpm=30, default_tpm=60000)
        pre_rpm_a = entry_a["rpm"]["mean"]
        pre_tpm_a = entry_a["tpm"]["mean"]

        target_a = t.learner.on_429(
            provider="prov_a",
            model="mod_a",
            estimated_tokens=1000,
            rolling_requests=28,
            rolling_tokens=1000,
            now=now,
            default_rpm=30,
            default_tpm=60000,
            target_dimension=None,  # generic 429
        )
        self.assertEqual(target_a, "rpm")
        self.assertLess(entry_a["rpm"]["mean"], pre_rpm_a)
        self.assertEqual(entry_a["tpm"]["mean"], pre_tpm_a)

        # Scenario B: Low RPM load (1/30 = 3.3%), high TPM load (58000/60000 = 96.6%)
        entry_b = t.learner.get_or_create_entry("prov_b", "mod_b", default_rpm=30, default_tpm=60000)
        pre_rpm_b = entry_b["rpm"]["mean"]
        pre_tpm_b = entry_b["tpm"]["mean"]

        target_b = t.learner.on_429(
            provider="prov_b",
            model="mod_b",
            estimated_tokens=8000,
            rolling_requests=1,
            rolling_tokens=50000,
            now=now,
            default_rpm=30,
            default_tpm=60000,
            target_dimension=None,  # generic 429
        )
        self.assertEqual(target_b, "tpm")
        self.assertEqual(entry_b["rpm"]["mean"], pre_rpm_b)
        self.assertLess(entry_b["tpm"]["mean"], pre_tpm_b)

    def test_configured_limits_are_absolute_policy_ceilings_and_particles_capped(self):
        ctx = DummyPluginContext(
            config={"requests_per_minute": 40, "tokens_per_minute": 80000}
        )
        t = GlobalThrottle(ctx)
        entry = t.learner.get_or_create_entry("provider_x", "model_x", default_rpm=40, default_tpm=80000)

        # Requirement 1 & 2: max_val is configured limit, not 5x
        self.assertEqual(entry["rpm"]["max_val"], 40.0)
        self.assertEqual(entry["tpm"]["max_val"], 80000.0)
        self.assertLessEqual(max(entry["rpm"]["particles"]), 40.0)
        self.assertLessEqual(max(entry["tpm"]["particles"]), 80000.0)

        # Even with high rolling traffic, effective limits never exceed configured ceiling
        now = time.time()
        for _ in range(20):
            t.learner.on_success(
                provider="provider_x",
                model="model_x",
                actual_tokens=10000,
                rolling_requests=100,
                rolling_tokens=200000,
                now=now,
                default_rpm=40,
                default_tpm=80000,
            )

        eff_rpm, eff_tpm = t.learner.get_effective_limits("provider_x", "model_x", default_rpm=40, default_tpm=80000)
        self.assertLessEqual(eff_rpm, 40)
        self.assertLessEqual(eff_tpm, 80000)
        self.assertEqual(eff_rpm, 40)
        self.assertEqual(eff_tpm, 80000)

    def test_429_lowers_limit_then_calibrates_in_spaced_steps(self):
        ctx = DummyPluginContext(
            config={"requests_per_minute": 50, "tokens_per_minute": 100000}
        )
        t = GlobalThrottle(ctx)
        now = time.time()

        # Step 1: Initial state has confidence 0, falls back to configured limits (Requirement 9)
        eff_rpm, eff_tpm = t.learner.get_effective_limits("p", "m", default_rpm=50, default_tpm=100000)
        self.assertEqual(eff_rpm, 50)
        self.assertEqual(eff_tpm, 100000)

        # Step 2: 429 response lowers the limit (Requirement 5)
        t.learner.on_429(
            provider="p",
            model="m",
            estimated_tokens=5000,
            rolling_requests=35,
            rolling_tokens=60000,
            now=now,
            default_rpm=50,
            default_tpm=100000,
            target_dimension="rpm",
        )
        entry = t.learner.get_or_create_entry("p", "m", default_rpm=50, default_tpm=100000)
        lowered_rpm = entry["safe_rpm"]
        self.assertLess(lowered_rpm, 50)
        # The learned rate stays effective throughout paced recalibration.
        self.assertLess(entry["confidence"], 0.70)
        fallback_rpm, _ = t.learner.get_effective_limits("p", "m", default_rpm=50, default_tpm=100000)
        self.assertEqual(fallback_rpm, lowered_rpm)

        # Step 3: Subsequent successful traffic increases confidence but does NOT recover upward (Requirements 3, 4, 6, 7)
        for i in range(20):
            t.learner.on_success(
                provider="p",
                model="m",
                actual_tokens=1000,
                rolling_requests=lowered_rpm,
                rolling_tokens=20000,
                now=now + i,
                default_rpm=50,
                default_tpm=100000,
            )
            # Ensure safe_rpm never recovers toward 50
            self.assertEqual(entry["safe_rpm"], lowered_rpm)

        # Step 4: High confidence reached at lower value -> lower value takes precedence (Requirement 8)
        self.assertGreaterEqual(entry["confidence"], 0.70)
        effective_rpm, _ = t.learner.get_effective_limits("p", "m", default_rpm=50, default_tpm=100000)
        self.assertEqual(effective_rpm, lowered_rpm)
        self.assertLess(effective_rpm, 50)

        # Step 5: A calibrating model advances only after the full success window.
        for i in range(30):
            t.learner.on_success(
                provider="p",
                model="m",
                actual_tokens=1000,
                rolling_requests=lowered_rpm,
                rolling_tokens=20000,
                now=now + 50 + i,
                default_rpm=50,
                default_tpm=100000,
            )
        self.assertGreater(entry["safe_rpm"], lowered_rpm)
        self.assertLess(entry["safe_rpm"], 50)
        effective_rpm, _ = t.learner.get_effective_limits("p", "m", default_rpm=50, default_tpm=100000)
        self.assertEqual(effective_rpm, entry["safe_rpm"])

        # Step 6: Deliberate manual reset discards lower ceiling and starts over (Requirement 10)
        self.assertTrue(t.learner.reset_learning("p", "m"))
        reset_rpm, reset_tpm = t.learner.get_effective_limits("p", "m", default_rpm=50, default_tpm=100000)
        self.assertEqual(reset_rpm, 50)
        self.assertEqual(reset_tpm, 100000)

    def test_confident_model_limit_becomes_a_persisted_safe_ceiling(self):
        ctx = DummyPluginContext(
            config={"requests_per_minute": 50, "tokens_per_minute": 100000}
        )
        throttle = GlobalThrottle(ctx)
        now = time.time()

        for offset in range(11):
            throttle.learner.on_success(
                provider="provider",
                model="model",
                actual_tokens=1000,
                rolling_requests=20,
                rolling_tokens=10000,
                now=now + offset + 1,
                default_rpm=50,
                default_tpm=100000,
            )

        entry = throttle.learner.get_or_create_entry(
            "provider", "model", default_rpm=50, default_tpm=100000
        )
        self.assertTrue(entry["safe_ceiling_established"])
        self.assertEqual(entry["established_safe_rpm"], entry["safe_rpm"])
        self.assertEqual(entry["established_safe_tpm"], entry["safe_tpm"])
        self.assertEqual(entry["established_observation_count"], entry["observation_count"])
        self.assertGreaterEqual(entry["established_confidence"], 0.70)

        with throttle._cv:
            throttle._persist_locked()
        persisted = ctx.state[PLUGIN_STATE_KEY]["learned_limits"]["provider::model"]
        self.assertTrue(persisted["safe_ceiling_established"])
        self.assertEqual(persisted["established_safe_rpm"], entry["safe_rpm"])

    def test_429_lowers_active_limit_without_lowering_established_ceiling(self):
        ctx = DummyPluginContext(
            config={"requests_per_minute": 50, "tokens_per_minute": 100000}
        )
        throttle = GlobalThrottle(ctx)
        now = time.time()

        for offset in range(11):
            throttle.learner.on_success(
                "provider", "model", 1000, 20, 10000, now + offset + 1, 50, 100000
            )
        entry = throttle.learner.get_or_create_entry("provider", "model", 50, 100000)
        first_ceiling = entry["established_safe_rpm"]

        throttle.learner.on_429(
            "provider", "model", 1000, 20, 10000, now + 20, 50, 100000,
            target_dimension="rpm",
        )

        self.assertEqual(entry["established_safe_rpm"], first_ceiling)
        self.assertLess(entry["safe_rpm"], first_ceiling)
        self.assertLess(entry["confidence"], 0.70)
        effective_rpm, _ = throttle.learner.get_effective_limits(
            "provider", "model", 50, 100000
        )
        self.assertEqual(effective_rpm, entry["safe_rpm"])
        self.assertLess(effective_rpm, 50)

    def test_established_model_does_not_recover_upward_after_a_429(self):
        learner = BayesianRateLimitLearner({})
        now = time.time()

        for offset in range(11):
            learner.on_success(
                "provider", "model", 1000, 20, 10000, now + offset + 1, 50, 100000
            )
        entry = learner.get_or_create_entry("provider", "model", 50, 100000)
        ceiling = entry["established_safe_rpm"]

        learner.on_429(
            "provider", "model", 1000, 20, 10000, now + 20, 50, 100000,
            target_dimension="rpm",
        )
        lowered = entry["safe_rpm"]
        self.assertLess(lowered, ceiling)

        for offset in range(100):
            learner.on_success(
                "provider", "model", 1000, lowered, 10000, now + 21 + offset, 50, 100000
            )
        effective_rpm, _ = learner.get_effective_limits(
            "provider", "model", 50, 100000
        )
        self.assertEqual(entry["safe_rpm"], lowered)
        self.assertEqual(effective_rpm, lowered)
        self.assertEqual(entry["established_safe_rpm"], ceiling)

    def test_established_model_does_not_advance_bucket_recovery_factor(self):
        throttle = GlobalThrottle(DummyPluginContext(
            config={"requests_per_minute": 50, "tokens_per_minute": 100000}
        ))
        now = time.time()
        for offset in range(11):
            throttle.learner.on_success(
                "provider", "model", 1000, 20, 10000, now + offset, 50, 100000
            )
        entry = throttle.learner.get_or_create_entry("provider", "model", 50, 100000)
        self.assertTrue(entry["safe_ceiling_established"])

        adaptive = throttle._global_bucket._state.setdefault("adaptive", {})
        adaptive["factor"] = 0.8
        adaptive["successes_since_429"] = 24
        throttle._global_bucket.on_success(
            "request", 1000, now + 20, provider="provider", model="model"
        )

        self.assertEqual(adaptive["factor"], 0.8)

    def test_calibrating_model_spaces_upward_safe_limit_steps(self):
        learner = BayesianRateLimitLearner({})
        now = time.time()
        learner.on_429(
            "provider", "model", 1000, 30, 10000, now, 50, 100000,
            target_dimension="rpm",
        )
        entry = learner.get_or_create_entry("provider", "model", 50, 100000)
        lowered = entry["safe_rpm"]

        for offset in range(24):
            learner.on_success(
                "provider", "model", 1000, lowered, 10000, now + offset + 1, 50, 100000
            )
        self.assertFalse(entry["safe_ceiling_established"])
        self.assertEqual(entry["safe_rpm"], lowered)

        learner.on_success(
            "provider", "model", 1000, lowered, 10000, now + 25, 50, 100000
        )
        first_step = entry["safe_rpm"]
        self.assertGreater(first_step, lowered)
        self.assertLess(first_step, 50)
        self.assertEqual(entry["successes_since_429"], 0)
        self.assertEqual(entry["confidence"], 0.0)
        effective_rpm, _ = learner.get_effective_limits("provider", "model", 50, 100000)
        self.assertEqual(effective_rpm, first_step)

        learner.on_success(
            "provider", "model", 1000, first_step, 10000, now + 26, 50, 100000
        )
        self.assertEqual(entry["safe_rpm"], first_step)
        self.assertEqual(entry["successes_since_429"], 1)

    def test_bucket_recovery_waits_for_the_next_calibration_window(self):
        throttle = GlobalThrottle(DummyPluginContext(
            config={"requests_per_minute": 50, "tokens_per_minute": 100000}
        ))
        learner = throttle.learner
        now = time.time()
        learner.on_429(
            "provider", "model", 1000, 30, 10000, now, 50, 100000,
            target_dimension="rpm",
        )
        for offset in range(25):
            learner.on_success(
                "provider", "model", 1000, 20, 10000, now + offset + 1, 50, 100000
            )
        entry = learner.get_or_create_entry("provider", "model", 50, 100000)
        self.assertEqual(entry["successes_since_429"], 0)

        adaptive = throttle._global_bucket._state.setdefault("adaptive", {})
        adaptive["factor"] = 0.8
        adaptive["successes_since_429"] = 24
        throttle._global_bucket.on_success(
            "request", 1000, now + 30, provider="provider", model="model"
        )

        self.assertEqual(adaptive["factor"], 0.8)

    def test_high_confidence_legacy_limit_is_migrated_to_an_established_ceiling(self):
        store = {}
        learner = BayesianRateLimitLearner(store)
        entry = learner.get_or_create_entry("provider", "model", 50, 100000)
        entry["safe_rpm"] = 30
        entry["rpm"]["safe_limit"] = 30.0
        entry["confidence"] = 0.8
        entry["observation_count"] = 20
        for key in (
            "safe_ceiling_established",
            "established_safe_rpm",
            "established_safe_tpm",
            "established_confidence",
            "established_observation_count",
            "safe_ceiling_established_at",
        ):
            entry.pop(key)

        migrated = learner.get_or_create_entry("provider", "model", 50, 100000)

        self.assertTrue(migrated["safe_ceiling_established"])
        self.assertEqual(migrated["established_safe_rpm"], 30)
        self.assertEqual(migrated["established_safe_tpm"], 100000)
        self.assertEqual(migrated["established_observation_count"], 20)

    def test_status_output_labels_configured_ceiling_learned_safe_limit_and_confidence(self):
        ctx = DummyPluginContext(
            config={"requests_per_minute": 25, "tokens_per_minute": 50000}
        )
        t = GlobalThrottle(ctx)
        t.learner.on_success("anthropic", "claude-3-5-sonnet", 2000, 10, 15000, time.time(), 25, 50000)
        with t._cv:
            t._persist_locked()

        status = t.handle_command(["status"])
        self.assertIn("Configured global limits: **25 RPM / 50,000 TPM / 1,000 RPD**", status)
        self.assertIn("| `anthropic::claude-3-5-sonnet` | 25 / 50,000 | — |", status)
        self.assertIn("Confidence", status)

    def test_status_shows_compact_learned_calibration_states_and_ceiling(self):
        ctx = DummyPluginContext(
            config={"requests_per_minute": 50, "tokens_per_minute": 100000}
        )
        throttle = GlobalThrottle(ctx)
        now = time.time()

        throttle.learner.on_success("calibrating", "model", 1000, 1, 1000, now, 50, 100000)
        throttle.learner.on_429(
            "recovering", "model", 1000, 30, 1000, now, 50, 100000,
            target_dimension="rpm",
        )
        for offset in range(11):
            throttle.learner.on_success(
                "established", "model", 1000, 1, 1000, now + offset, 50, 100000
            )
        with throttle._cv:
            throttle._persist_locked()

        status = throttle.handle_command(["status"])

        self.assertIn("| Scope | Current safe (RPM / TPM) | Established ceiling (RPM / TPM) |", status)
        self.assertIn("| `calibrating::model` | 50 / 100,000 | — |", status)
        self.assertIn("| `recovering::model` |", status)
        self.assertIn("| `established::model` | 50 / 100,000 | 50 / 100,000 |", status)
        self.assertIn("| calibrating |", status)
        self.assertIn("| recovering |", status)
        self.assertIn("| established |", status)


class ScopedThrottleTests(unittest.TestCase):
    def setUp(self):
        self.ctx = DummyPluginContext(
            config={
                "enabled": True,
                "requests_per_minute": 30,
                "tokens_per_minute": 60000,
                "requests_per_day": 1000,
                "make_daily_quota_last_hours": 8.0,
                "longest_wait_between_requests": 15.0,
            }
        )
        self.throttle = GlobalThrottle(self.ctx)

    def test_default_global_bucket_used_when_no_override_exists(self):
        # 1. Keep the current global bucket as the default when no override exists.
        req_id = "req_global_1"
        self.throttle.on_pre_api_request(
            request_id=req_id,
            provider="unknown_provider",
            model="unknown_model",
            approx_input_tokens=100,
        )
        result = self.throttle.wrap_llm_execution(
            request={"provider": "unknown_provider", "model": "unknown_model"},
            next_call=lambda r: "ok",
            request_id=req_id,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(self.throttle.request_count, 1)
        self.assertEqual(len(self.throttle.reservations), 1)

        # Complete request
        self.throttle.on_post_api_request(
            request_id=req_id,
            provider="unknown_provider",
            model="unknown_model",
            usage={"total_tokens": 150},
        )
        self.assertEqual(len(self.throttle.reservations), 0)
        self.assertEqual(len(self.throttle.token_ledger), 1)
        self.assertEqual(self.throttle.token_ledger[0][1], 150)

    def test_provider_level_override(self):
        # 2. Add optional provider-level RPM/TPM/RPD overrides.
        self.ctx.config["overrides"] = {
            "anthropic": {"rpm": 60, "tpm": 120000, "rpd": 500}
        }
        throttle = GlobalThrottle(self.ctx)

        req_id = "req_ant_1"
        throttle.on_pre_api_request(
            request_id=req_id,
            provider="anthropic",
            model="claude-3-opus",
            approx_input_tokens=100,
        )
        throttle.wrap_llm_execution(
            request={"provider": "anthropic", "model": "claude-3-opus"},
            next_call=lambda r: "ok",
            request_id=req_id,
            provider="anthropic",
            model="claude-3-opus",
        )

        ant_bucket = throttle.get_bucket("anthropic")
        self.assertEqual(ant_bucket.request_count, 1)
        self.assertEqual(throttle.request_count, 0)  # global bucket untouched

        # Post response
        throttle.on_post_api_request(
            request_id=req_id,
            provider="anthropic",
            model="claude-3-opus",
            usage={"total_tokens": 250},
        )
        self.assertEqual(len(ant_bucket.token_ledger), 1)
        self.assertEqual(ant_bucket.token_ledger[0][1], 250)
        self.assertEqual(len(throttle.token_ledger), 0)

    def test_model_level_override(self):
        # 3. Add optional provider::model RPM/TPM/RPD overrides.
        self.ctx.config["overrides"] = {
            "openai::gpt-4o": {"rpm": 120, "tpm": 240000, "rpd": 2000}
        }
        throttle = GlobalThrottle(self.ctx)

        req_id = "req_gpt_1"
        throttle.on_pre_api_request(
            request_id=req_id,
            provider="openai",
            model="gpt-4o",
            approx_input_tokens=100,
        )
        throttle.wrap_llm_execution(
            request={"provider": "openai", "model": "gpt-4o"},
            next_call=lambda r: "ok",
            request_id=req_id,
            provider="openai",
            model="gpt-4o",
        )

        mod_bucket = throttle.get_bucket("openai::gpt-4o")
        self.assertEqual(mod_bucket.request_count, 1)
        self.assertEqual(throttle.request_count, 0)

        throttle.on_post_api_request(
            request_id=req_id,
            provider="openai",
            model="gpt-4o",
            usage={"total_tokens": 300},
        )
        self.assertEqual(len(mod_bucket.token_ledger), 1)
        self.assertEqual(mod_bucket.token_ledger[0][1], 300)

    def test_limit_resolution_hierarchy(self):
        # 4. Resolve limits in this order: model override -> provider override -> global default.
        # Global default: RPM=30, TPM=60000, RPD=1000
        self.ctx.config["overrides"] = {
            "openai": {"rpm": 60, "tpm": 100000},  # RPD not specified
            "openai::gpt-4o": {"rpm": 120},  # TPM and RPD not specified
        }
        throttle = GlobalThrottle(self.ctx)

        # Model override resolves: RPM from model (120), TPM from provider (100000), RPD from global (1000)
        gpt4o_limits = throttle.resolve_limits("openai", "gpt-4o")
        self.assertEqual(gpt4o_limits.rpm, 120)
        self.assertEqual(gpt4o_limits.tpm, 100000)
        self.assertEqual(gpt4o_limits.rpd, 1000)

        # Provider override resolves: RPM from provider (60), TPM from provider (100000), RPD from global (1000)
        openai_limits = throttle.resolve_limits("openai", "other-model")
        self.assertEqual(openai_limits.rpm, 60)
        self.assertEqual(openai_limits.tpm, 100000)
        self.assertEqual(openai_limits.rpd, 1000)

        # Other provider with no override resolves all from global default
        anthropic_limits = throttle.resolve_limits("anthropic", "claude")
        self.assertEqual(anthropic_limits.rpm, 30)
        self.assertEqual(anthropic_limits.tpm, 60000)
        self.assertEqual(anthropic_limits.rpd, 1000)

    def test_per_scope_rpm_tpm_reservations_and_pacing(self):
        # 5. Give each configured provider/model scope its own RPM state, TPM ledger,
        # reservations, and pacing state.
        self.ctx.config["overrides"] = {
            "prov_a": {"rpm": 60, "tpm": 60000},
            "prov_b": {"rpm": 120, "tpm": 120000},
        }
        throttle = GlobalThrottle(self.ctx)
        bucket_a = throttle.get_bucket("prov_a")
        bucket_b = throttle.get_bucket("prov_b")

        # Dispatch to prov_a
        throttle.wrap_llm_execution(
            request={"provider": "prov_a", "model": "m"},
            next_call=lambda r: "a",
            request_id="req_a",
            provider="prov_a",
            model="m",
        )
        self.assertEqual(bucket_a.request_count, 1)
        self.assertEqual(bucket_b.request_count, 0)
        self.assertGreater(bucket_a._last_dispatch_mono, 0.0)
        self.assertEqual(bucket_b._last_dispatch_mono, 0.0)

        # prov_a post API request
        throttle.on_post_api_request(
            request_id="req_a",
            provider="prov_a",
            model="m",
            usage={"total_tokens": 500},
        )
        self.assertEqual(len(bucket_a.token_ledger), 1)
        self.assertEqual(len(bucket_b.token_ledger), 0)

    def test_both_provider_and_model_limits_apply_and_satisfy_both(self):
        # 6. If both provider and model limits apply, require the request to satisfy both.
        self.ctx.config["overrides"] = {
            "openai": {"rpm": 60, "tpm": 100000},
            "openai::slow-model": {"rpm": 6, "tpm": 10000},
        }
        throttle = GlobalThrottle(self.ctx)
        b_prov = throttle.get_bucket("openai")
        b_mod = throttle.get_bucket("openai::slow-model")

        req_id = "req_both_1"
        throttle.on_pre_api_request(
            request_id=req_id,
            provider="openai",
            model="slow-model",
            approx_input_tokens=100,
        )

        throttle.wrap_llm_execution(
            request={"provider": "openai", "model": "slow-model"},
            next_call=lambda r: "done",
            request_id=req_id,
            provider="openai",
            model="slow-model",
        )

        # Both buckets must record dispatch and reservations
        self.assertEqual(b_prov.request_count, 1)
        self.assertEqual(b_mod.request_count, 1)
        self.assertIn(req_id, b_prov.reservations)
        self.assertIn(req_id, b_mod.reservations)

        # Complete request
        throttle.on_post_api_request(
            request_id=req_id,
            provider="openai",
            model="slow-model",
            usage={"total_tokens": 400},
        )
        self.assertNotIn(req_id, b_prov.reservations)
        self.assertNotIn(req_id, b_mod.reservations)
        self.assertEqual(len(b_prov.token_ledger), 1)
        self.assertEqual(len(b_mod.token_ledger), 1)
        self.assertEqual(b_prov.token_ledger[0][1], 400)
        self.assertEqual(b_mod.token_ledger[0][1], 400)

    def test_blocked_bucket_does_not_block_unrelated_buckets(self):
        # 7. Prevent one blocked bucket from blocking unrelated buckets.
        self.ctx.config["overrides"] = {
            "anthropic": {"rpm": 60, "tpm": 60000, "rpd": 1},
            "openai": {"rpm": 60, "tpm": 60000, "rpd": 1000},
        }
        throttle = GlobalThrottle(self.ctx)
        ant_bucket = throttle.get_bucket("anthropic")
        openai_bucket = throttle.get_bucket("openai")

        # Trigger 429 on anthropic to put it in a long cooldown
        throttle.on_api_request_error(
            request_id="req_ant_err",
            provider="anthropic",
            model="claude-3-opus",
            status_code=429,
        )
        self.assertGreater(ant_bucket.cooldown_until, time.time() + 5.0)
        self.assertEqual(openai_bucket.cooldown_until, 0.0)
        self.assertEqual(throttle.cooldown_until, 0.0)

        # Request to openai dispatches immediately without being blocked by anthropic
        t_start = time.time()
        res = throttle.wrap_llm_execution(
            request={"provider": "openai", "model": "gpt-4o"},
            next_call=lambda r: "openai_ok",
            request_id="req_oai_1",
            provider="openai",
            model="gpt-4o",
        )
        elapsed = time.time() - t_start
        self.assertEqual(res, "openai_ok")
        self.assertLess(elapsed, 0.5)

        # Unrelated global request also dispatches immediately
        t_start = time.time()
        res_global = throttle.wrap_llm_execution(
            request={"provider": "google", "model": "gemini-1.5-pro"},
            next_call=lambda r: "global_ok",
            request_id="req_glob_1",
            provider="google",
            model="gemini-1.5-pro",
        )
        elapsed_global = time.time() - t_start
        self.assertEqual(res_global, "global_ok")
        self.assertLess(elapsed_global, 0.5)

    def test_fifo_preservation_within_each_bucket(self):
        # 8. Preserve FIFO within each bucket.
        self.ctx.config["overrides"] = {
            "fast_provider": {"rpm": 6000, "tpm": 600000, "hours": 0.0, "max_wait": 0.0},
        }
        throttle = GlobalThrottle(self.ctx)
        dispatch_order = []

        def worker(req_id):
            throttle.wrap_llm_execution(
                request={"provider": "fast_provider", "model": "m"},
                next_call=lambda r: dispatch_order.append(req_id),
                request_id=req_id,
                provider="fast_provider",
                model="m",
            )

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(f"req_{i}",))
            threads.append(t)

        for t in threads:
            t.start()
            time.sleep(0.01)  # small stagger to guarantee deterministic arrival order
        for t in threads:
            t.join()

        self.assertEqual(dispatch_order, [f"req_{i}" for i in range(5)])

    def test_commands_and_config_for_overrides(self):
        # 9. Add commands/config for setting, viewing, and removing provider/model overrides.
        # Initially empty
        res = self.throttle.handle_command(["overrides"])
        self.assertIn("No provider or model overrides configured", res)

        # Set provider override
        res = self.throttle.handle_command(["set", "openai", "rpm", "60", "tpm", "100000"])
        self.assertIn("Set override for openai", res)
        self.assertIn("RPM=60", res)
        self.assertIn("TPM=100000", res)

        # Set model override with provider::model syntax
        res = self.throttle.handle_command(["set", "openai::gpt-4o", "rpm", "120", "rpd", "500"])
        self.assertIn("Set override for openai::gpt-4o", res)

        # Set model override with space separation: <provider> <model>
        res = self.throttle.handle_command(["set", "anthropic", "claude-3-opus", "rpm", "15"])
        self.assertIn("Set override for anthropic::claude-3-opus", res)

        # View all overrides
        res = self.throttle.handle_command(["overrides"])
        self.assertIn("Configured Overrides:", res)
        self.assertIn("openai", res)
        self.assertIn("openai::gpt-4o", res)
        self.assertIn("anthropic::claude-3-opus", res)

        # View specific override
        res = self.throttle.handle_command(["get", "openai"])
        self.assertIn("Override for openai", res)
        self.assertIn("resolved limits: 60 RPM | 100000 TPM", res)

        # Remove parameter from override
        res = self.throttle.handle_command(["remove", "openai", "tpm"])
        self.assertIn("Removed TPM override for openai", res)
        override = self.throttle.get_override("openai")
        self.assertNotIn("tpm", override)
        self.assertIn("rpm", override)

        # Remove entire override
        res = self.throttle.handle_command(["remove", "openai"])
        self.assertIn("Override removed for openai", res)
        self.assertIsNone(self.throttle.get_override("openai"))

        # Remove non-existent override
        res = self.throttle.handle_command(["remove", "openai"])
        self.assertIn("No override found for openai", res)

    def test_status_text_shows_global_and_active_scoped_buckets(self):
        # 10. Update /throttle status to show global and active scoped buckets.
        # When no active scoped buckets exist
        res = self.throttle.handle_command(["status"])
        self.assertIn("Global Throttle Status", res)
        self.assertNotIn("### Active scoped buckets", res)

        # Add an override and execute a request
        self.throttle.handle_command(["set", "openai", "rpm", "60", "tpm", "100000"])
        self.throttle.wrap_llm_execution(
            request={"provider": "openai", "model": "gpt-4o"},
            next_call=lambda r: "ok",
            request_id="req_status_test",
            provider="openai",
            model="gpt-4o",
        )

        res = self.throttle.handle_command(["status"])
        self.assertIn("Global Throttle Status", res)
        self.assertIn("### Active scoped buckets", res)
        self.assertIn("| `openai` | provider | 60 RPM / 100,000 TPM / 1,000 RPD |", res)
        self.assertIn("| Queue | Pacing | Adaptive | Reason |", res)

    def test_provider_more_restrictive_than_model(self):
        # When provider limit is more restrictive than model limit, provider limit governs
        self.ctx.config["overrides"] = {
            "openai": {"rpm": 10, "tpm": 100000, "hours": 0.0, "max_wait": 0.0},
            "openai::fast": {"rpm": 60, "tpm": 100000, "hours": 0.0, "max_wait": 0.0},
        }
        throttle = GlobalThrottle(self.ctx)
        b_prov = throttle.get_bucket("openai")
        b_mod = throttle.get_bucket("openai::fast")

        # First request dispatches
        throttle.wrap_llm_execution(
            request={"provider": "openai", "model": "fast"},
            next_call=lambda r: "ok",
            request_id="req1",
            provider="openai",
            model="fast",
        )

        # Provider bucket was dispatched at t_now, so provider requires 60/10 = 6.0s spacing
        wait_for, reason, _ = b_prov._check_wait_locked("req2", 100, "openai", "fast")
        self.assertAlmostEqual(wait_for, 6.0, delta=0.2)
        self.assertEqual(reason, "RPM")

        # Model bucket requires 60/60 = 1.0s spacing
        wait_mod, reason_mod, _ = b_mod._check_wait_locked("req2", 100, "openai", "fast")
        self.assertAlmostEqual(wait_mod, 1.0, delta=0.2)

    def test_idle_burst_credit_is_capped_at_two_dispatches_per_model(self):
        self.ctx.config.update({
            "requests_per_minute": 60,
            "tokens_per_minute": 600,
            "make_daily_quota_last_hours": 0.0,
            "longest_wait_between_requests": 0.0,
        })
        throttle = GlobalThrottle(self.ctx)
        bucket = throttle.get_applicable_buckets("provider", "model")[0]

        with patch("throttle.time.monotonic", return_value=1000.0):
            wait, _, _ = bucket._check_wait_locked("first", 20, "provider", "model")
            self.assertEqual(wait, 0.0)
            bucket._record_dispatch_locked("first", 20, "provider", "model")
            wait, _, _ = bucket._check_wait_locked("second", 20, "provider", "model")
            self.assertAlmostEqual(wait, 2.0)

        with patch("throttle.time.monotonic", return_value=1010.0):
            wait, _, _ = bucket._check_wait_locked("second", 20, "provider", "model")
            self.assertEqual(wait, 0.0)
            bucket._record_dispatch_locked("second", 20, "provider", "model")
            wait, _, _ = bucket._check_wait_locked("third", 20, "provider", "model")
            self.assertEqual(wait, 0.0)
            bucket._record_dispatch_locked("third", 20, "provider", "model")
            wait, _, _ = bucket._check_wait_locked("fourth", 20, "provider", "model")
            self.assertAlmostEqual(wait, 2.0)

    def test_idle_burst_credit_is_separate_for_models_in_a_provider_bucket(self):
        self.ctx.config.update({
            "overrides": {"provider": {"rpm": 60}},
            "make_daily_quota_last_hours": 0.0,
            "longest_wait_between_requests": 0.0,
        })
        throttle = GlobalThrottle(self.ctx)
        bucket = throttle.get_bucket("provider")

        with patch("throttle.time.monotonic", return_value=1000.0):
            bucket._check_wait_locked("a1", 20, "provider", "model-a")
            bucket._record_dispatch_locked("a1", 20, "provider", "model-a")
            wait_a, _, _ = bucket._check_wait_locked("a2", 20, "provider", "model-a")
            wait_b, _, _ = bucket._check_wait_locked("b1", 20, "provider", "model-b")

        self.assertAlmostEqual(wait_a, 1.0)
        self.assertEqual(wait_b, 0.0)

    def test_idle_burst_credit_does_not_override_tpm_ceiling_or_backoff(self):
        self.ctx.config.update({
            "requests_per_minute": 60,
            "tokens_per_minute": 600,
            "make_daily_quota_last_hours": 0.0,
            "longest_wait_between_requests": 0.0,
        })
        throttle = GlobalThrottle(self.ctx)
        bucket = throttle.get_applicable_buckets("provider", "model")[0]
        bucket._burst_credit["provider::model"] = (2.0, time.monotonic())
        bucket._token_ledger.append((time.time(), 550))

        wait, reason, _ = bucket._check_wait_locked("request", 100, "provider", "model")
        self.assertGreater(wait, 0.0)
        self.assertEqual(reason, "TPM ceiling")

        bucket._token_ledger.clear()
        bucket._state.setdefault("adaptive", {})["cooldown_until"] = time.time() + 10.0
        wait, reason, _ = bucket._check_wait_locked("request", 100, "provider", "model")
        self.assertGreater(wait, 9.0)
        self.assertEqual(reason, "429 cooldown")

        bucket._state["adaptive"]["cooldown_until"] = 0.0
        learned = throttle.learner.get_or_create_entry("provider", "model", 60, 600)
        learned["slowdown_until"] = time.time() + 12.0
        wait, reason, _ = bucket._check_wait_locked("request", 100, "provider", "model")
        self.assertGreater(wait, 11.0)
        self.assertEqual(reason, "learned slowdown")

        learned["slowdown_until"] = 0.0
        bucket._state["adaptive"]["factor"] = 0.8
        bucket._last_dispatch_mono = time.monotonic()
        wait, _, _ = bucket._check_wait_locked("request", 100, "provider", "model")
        self.assertGreater(wait, 12.0)

    def test_idle_burst_credit_keeps_learned_tpm_and_rpd_gates(self):
        self.ctx.config.update({
            "requests_per_minute": 60,
            "tokens_per_minute": 600,
            "make_daily_quota_last_hours": 1.0,
            "longest_wait_between_requests": 15.0,
        })
        throttle = GlobalThrottle(self.ctx)
        bucket = throttle.get_applicable_buckets("provider", "model")[0]
        bucket._burst_credit["provider::model"] = (2.0, time.monotonic())
        learned = throttle.learner.get_or_create_entry("provider", "model", 60, 600)
        learned["safe_tpm"] = 100
        learned["confidence"] = 1.0
        bucket._token_ledger.append((time.time(), 80))

        wait, reason, _ = bucket._check_wait_locked("request", 30, "provider", "model")
        self.assertGreater(wait, 0.0)
        self.assertEqual(reason, "TPM ceiling")

        bucket._token_ledger.clear()
        scope = throttle._get_rpd_scope_entry_locked("provider::model", "provider", "model")
        scope["timestamps"] = [time.time()]
        scope["first_request_at"] = time.time()
        scope["last_dispatch_mono"] = time.monotonic()
        wait, _, _ = bucket._check_wait_locked("request", 30, "provider", "model")
        self.assertAlmostEqual(wait, 3600.0 / 999.0, delta=0.1)

    def test_two_models_under_same_provider_do_not_block_each_other(self):
        # 7. Prevent one blocked bucket from blocking unrelated buckets.
        # If model A under provider P is blocked by model A's rate limit,
        # model B under provider P must NOT be blocked by model A's rate limit!
        self.ctx.config["overrides"] = {
            "openai": {"rpm": 1200, "tpm": 600000, "hours": 0.0, "max_wait": 0.0},
            "openai::slow": {"rpm": 1, "tpm": 10000, "hours": 0.0, "max_wait": 0.0},  # 60s spacing
            "openai::fast": {"rpm": 1200, "tpm": 600000, "hours": 0.0, "max_wait": 0.0},
        }
        throttle = GlobalThrottle(self.ctx)

        # Dispatch first request to slow model
        throttle.wrap_llm_execution(
            request={"provider": "openai", "model": "slow"},
            next_call=lambda r: "slow_1",
            request_id="req_slow_1",
            provider="openai",
            model="slow",
        )

        # openai::slow is now blocked for ~60s
        b_slow = throttle.get_bucket("openai::slow")
        wait_slow, _, _ = b_slow._check_wait_locked("req_slow_2", 100, "openai", "slow")
        self.assertGreater(wait_slow, 50.0)

        # But openai::fast is completely unblocked and dispatches in < 0.5s without waiting 60s
        t_start = time.time()
        res = throttle.wrap_llm_execution(
            request={"provider": "openai", "model": "fast"},
            next_call=lambda r: "fast_ok",
            request_id="req_fast_1",
            provider="openai",
            model="fast",
        )
        elapsed = time.time() - t_start
        self.assertEqual(res, "fast_ok")
        self.assertLess(elapsed, 0.5)

    def test_commands_key_value_and_validation(self):
        # Test key=value syntax in set command
        res = self.throttle.handle_command(["set", "openai", "rpm=45", "tpm=90000"])
        self.assertIn("Set override for openai", res)
        self.assertIn("RPM=45", res)
        self.assertIn("TPM=90000", res)
        override = self.throttle.get_override("openai")
        self.assertEqual(override.get("rpm"), 45)
        self.assertEqual(override.get("tpm"), 90000)

        # Negative value validation
        res_neg = self.throttle.handle_command(["set", "openai", "rpm", "-5"])
        self.assertIn("must be greater than zero", res_neg)

        # Invalid value validation
        res_inv = self.throttle.handle_command(["set", "openai", "rpm", "not-a-number"])
        self.assertIn("Invalid RPM value", res_inv)

        # Unknown parameter
        res_unk = self.throttle.handle_command(["set", "openai", "foo", "123"])
        self.assertIn("Unknown parameter", res_unk)

        # Missing value
        res_mis = self.throttle.handle_command(["set", "openai", "rpm"])
        self.assertIn("Missing value for parameter", res_mis)

        # Empty set
        res_empty = self.throttle.handle_command(["set"])
        self.assertIn("Usage:", res_empty)

        # Get non-existent
        res_get_none = self.throttle.handle_command(["get", "non_existent_provider"])
        self.assertIn("No override configured for non_existent_provider", res_get_none)

    def test_exception_in_next_call_releases_reservations_in_both_buckets(self):
        self.ctx.config["overrides"] = {
            "openai": {"rpm": 60, "tpm": 100000},
            "openai::gpt-4": {"rpm": 30, "tpm": 50000},
        }
        throttle = GlobalThrottle(self.ctx)
        b_prov = throttle.get_bucket("openai")
        b_mod = throttle.get_bucket("openai::gpt-4")

        def failing_call(r):
            raise RuntimeError("provider API exploded")

        with self.assertRaises(RuntimeError):
            throttle.wrap_llm_execution(
                request={"provider": "openai", "model": "gpt-4"},
                next_call=failing_call,
                request_id="req_fail",
                provider="openai",
                model="gpt-4",
            )

        # Verify reservations were released in both buckets
        self.assertNotIn("req_fail", b_prov.reservations)
        self.assertNotIn("req_fail", b_mod.reservations)

    def test_scoped_rpd_accounting_by_provider_model(self):
        # 2 & 3: Scoped RPD counters by provider::model; requests from unrelated providers do not count against Gemini
        ctx = DummyPluginContext(
            config={
                "enabled": True,
                "requests_per_minute": 60000,
                "tokens_per_minute": 6000000,
                "requests_per_day": 1000,
                "make_daily_quota_last_hours": 0.0,
                "longest_wait_between_requests": 0.0,
            }
        )
        throttle = GlobalThrottle(ctx)

        # Dispatch 1 request for Gemini
        throttle.wrap_llm_execution(
            request={"provider": "google", "model": "gemini-1.5-pro"},
            next_call=lambda r: "gemini_ok",
            request_id="req_g_1",
            provider="google",
            model="gemini-1.5-pro",
        )
        # Dispatch 3 requests for OpenAI
        for i in range(3):
            throttle.wrap_llm_execution(
                request={"provider": "openai", "model": "gpt-4o"},
                next_call=lambda r: "oai_ok",
                request_id=f"req_o_{i}",
                provider="openai",
                model="gpt-4o",
            )
        # Dispatch 1 request for Anthropic
        throttle.wrap_llm_execution(
            request={"provider": "anthropic", "model": "claude-3-opus"},
            next_call=lambda r: "ant_ok",
            request_id="req_a_1",
            provider="anthropic",
            model="claude-3-opus",
        )

        g_count, _ = throttle.get_rpd_usage("google::gemini-1.5-pro")
        o_count, _ = throttle.get_rpd_usage("openai::gpt-4o")
        a_count, _ = throttle.get_rpd_usage("anthropic::claude-3-opus")

        # Gemini count must be exactly 1, unaffected by OpenAI or Anthropic
        self.assertEqual(g_count, 1)
        self.assertEqual(o_count, 3)
        self.assertEqual(a_count, 1)

    def test_gemini_500_default_limit_and_overrides(self):
        # 4: Gemini's 500 limit applies only to Gemini scope; non-Gemini gets configured RPD
        throttle = GlobalThrottle(self.ctx)

        limits_gemini_pro = throttle.resolve_limits(provider="google", model="gemini-1.5-pro")
        self.assertEqual(limits_gemini_pro.rpd, 500)

        limits_gemini_flash = throttle.resolve_limits(provider="google", model="gemini-1.5-flash")
        self.assertEqual(limits_gemini_flash.rpd, 500)

        limits_gemini_direct = throttle.resolve_limits(provider="gemini", model="gemini-2.0-flash")
        self.assertEqual(limits_gemini_direct.rpd, 500)

        # Non-Gemini gets global default (1000)
        limits_openai = throttle.resolve_limits(provider="openai", model="gpt-4o")
        self.assertEqual(limits_openai.rpd, 1000)

        # Explicit override on Gemini model takes precedence
        throttle.set_override("google::gemini-1.5-pro", rpd=1500)
        limits_gemini_overridden = throttle.resolve_limits(provider="google", model="gemini-1.5-pro")
        self.assertEqual(limits_gemini_overridden.rpd, 1500)

        # Other Gemini models still default to 500
        limits_other_gemini = throttle.resolve_limits(provider="google", model="gemini-1.5-flash")
        self.assertEqual(limits_other_gemini.rpd, 500)

    def test_rpd_rolling_window_continuous_rolloff(self):
        # 4: Actual rolling-window behavior (24h continuous roll-off, not calendar-day shutdown)
        throttle = GlobalThrottle(self.ctx)
        now = time.time()

        # Seed timestamps: 3 from 25 hours ago (expired), 2 from 2 hours ago (active), 1 now
        entry = throttle._get_rpd_scope_entry_locked("test::rolling-model", "test", "rolling-model")
        entry["timestamps"] = [
            now - 26 * 3600.0,
            now - 25.5 * 3600.0,
            now - 25.0 * 3600.0,
            now - 2.0 * 3600.0,
            now - 1.0 * 3600.0,
            now,
        ]

        count, time_until_rolloff = throttle.get_rpd_usage("test::rolling-model", now)
        # Expired 25+ hour old requests must have rolled off; only 3 within rolling 24h remain
        self.assertEqual(count, 3)
        self.assertIsNotNone(time_until_rolloff)
        # Oldest active request was 2 hours ago, so it rolls off in ~22 hours
        self.assertAlmostEqual(time_until_rolloff, 22.0 * 3600.0, delta=10.0)

    def test_rpd_pacing_increases_spacing_without_blocking_indefinitely(self):
        # 5 & 6: Pacing target approaching/exceeded increases spacing rather than stopping requests
        throttle = GlobalThrottle(self.ctx)
        now = time.time()

        # Fill Gemini scope with 500 timestamps within the 24h window
        entry = throttle._get_rpd_scope_entry_locked("google::gemini-1.5-pro", "google", "gemini-1.5-pro")
        entry["timestamps"] = [now - 100.0 + i * 0.1 for i in range(500)]
        entry["first_request_at"] = now - 100.0

        settings = throttle.resolve_limits(provider="google", model="gemini-1.5-pro")
        self.assertEqual(settings.rpd, 500)

        # Compute RPD pacing: must return increased spacing (capped and safe), NOT hours until midnight
        rpd_delay, raw_rpd_delay, count, limit = throttle.compute_rpd_pacing_locked(
            scope_key="google::gemini-1.5-pro",
            settings=settings,
            now=now,
            provider="google",
            model="gemini-1.5-pro",
        )
        self.assertEqual(count, 500)
        self.assertEqual(limit, 500)
        # Increased spacing must be bounded (e.g. 15s - 30s)
        self.assertGreaterEqual(rpd_delay, 15.0)
        self.assertLessEqual(rpd_delay, 30.0)

        # Set last_dispatch_mono into the past so the pacing interval has already elapsed
        with throttle._cv:
            entry["last_dispatch_mono"] = time.monotonic() - 35.0
            gemini_bucket = throttle.get_bucket("google::gemini-1.5-pro")
            gemini_bucket._last_dispatch_mono = time.monotonic() - 35.0

        # Dispatch must succeed cleanly without hanging or being killed by watchdog
        t_start = time.time()
        res = throttle.wrap_llm_execution(
            request={"provider": "google", "model": "gemini-1.5-pro"},
            next_call=lambda r: "gemini_dispatched_cleanly",
            request_id="req_g_501",
            provider="google",
            model="gemini-1.5-pro",
        )
        elapsed = time.time() - t_start
        self.assertEqual(res, "gemini_dispatched_cleanly")
        self.assertLess(elapsed, 0.5)

    def test_exhausted_gemini_scope_does_not_block_unrelated_providers(self):
        # 5 & 6: Other providers/models must continue normally when Gemini scope is pacing
        throttle = GlobalThrottle(self.ctx)
        now = time.time()

        # Seed Gemini scope to 500 requests (quota reached)
        g_entry = throttle._get_rpd_scope_entry_locked("google::gemini-1.5-pro", "google", "gemini-1.5-pro")
        g_entry["timestamps"] = [now - 50.0 + i * 0.1 for i in range(500)]
        g_entry["last_dispatch_mono"] = time.monotonic()

        # Gemini bucket is pacing (would wait ~15s)
        g_bucket = throttle.get_bucket("google::gemini-1.5-pro")
        g_bucket._last_dispatch_mono = time.monotonic()
        wait_g, reason_g, _ = g_bucket._check_wait_locked("req_g_test", 100, "google", "gemini-1.5-pro")
        self.assertGreater(wait_g, 10.0)
        self.assertEqual(reason_g, "RPD")

        # Meanwhile, an OpenAI request arrives: must dispatch immediately without waiting!
        t_start = time.time()
        res = throttle.wrap_llm_execution(
            request={"provider": "openai", "model": "gpt-4o"},
            next_call=lambda r: "openai_unblocked",
            request_id="req_oai_free",
            provider="openai",
            model="gpt-4o",
        )
        elapsed = time.time() - t_start
        self.assertEqual(res, "openai_unblocked")
        self.assertLess(elapsed, 0.5)

    def test_throttle_status_shows_scoped_rpd_separately_not_misleading_global(self):
        # 7: Update throttle status so it shows each scope's RPD usage/window separately
        ctx = DummyPluginContext(
            config={
                "enabled": True,
                "requests_per_minute": 60000,
                "tokens_per_minute": 6000000,
                "requests_per_day": 1000,
                "make_daily_quota_last_hours": 0.0,
                "longest_wait_between_requests": 0.0,
            }
        )
        throttle = GlobalThrottle(ctx)

        # Dispatch requests to both Gemini and OpenAI
        throttle.wrap_llm_execution(
            request={"provider": "google", "model": "gemini-1.5-pro"},
            next_call=lambda r: "ok",
            request_id="req_stat_g",
            provider="google",
            model="gemini-1.5-pro",
        )
        for i in range(2):
            throttle.wrap_llm_execution(
                request={"provider": "openai", "model": "gpt-4o"},
                next_call=lambda r: "ok",
                request_id=f"req_stat_o_{i}",
                provider="openai",
                model="gpt-4o",
            )

        status_text = throttle.handle_command(["status"])
        self.assertIn("Global Throttle Status", status_text)
        self.assertIn("### Scoped rolling RPD usage", status_text)
        self.assertIn("| `google::gemini-1.5-pro` | 1 / 500 |", status_text)
        self.assertIn("| `openai::gpt-4o` | 2 / 1,000 |", status_text)


if __name__ == "__main__":
    unittest.main()
