# Hermes Global Throttle

A deliberately simple global leaky-bucket throttle for Hermes Agent.

## What it does

Every outbound LLM provider attempt passes through one FIFO queue. The plugin smooths launches so Hermes does not machine-gun provider APIs.

The v1 pacing model is:

```text
RPM delay = 60 / RPM
TPM delay = 60 * estimated_tokens_per_request / TPM

RPD raw delay =
    remaining desired usage time
    / remaining daily requests

RPD delay = min(RPD raw delay, longest_wait_between_requests)

actual spacing = max(RPM delay, TPM delay, RPD delay)
```

RPM and TPM are hard pacing constraints.

RPD does two jobs:

1. It *softly* tries to make the daily allowance last for the configured number of hours.
2. When `requests_per_day` is actually exhausted, it waits for the next local calendar day.

The `longest_wait_between_requests` cap applies only to RPD preservation. It never overrides RPM or TPM.

Each `provider::model` bucket can accumulate at most two requests worth of
idle-time burst credit. Credit may reduce RPM/TPM spacing after idle periods,
but it never bypasses the rolling TPM ledger, learned safe limits, cooldowns,
RPD pacing, or 429 backoff.

## Global weighted capacity pool

All work entering the active LLM execution middleware also passes through one
global weighted capacity pool. New work waits FIFO when the pool is full;
active work is never cancelled. If capacity is reduced below current usage,
the pool becomes temporarily overcommitted and waits for running work to finish
naturally before admitting anything new. Rough default weights are:

`interactive=1`, `background=2`, `tool_call=3`, `other=3`, `api_request=4`,
`llm_api=5`, and `subagent=8`.

The default pool capacity is 100 units and can be changed with
`capacity_units`; individual weights can be overridden with `workload_weights`.
This pool is separate from provider RPM/TPM limiting. Remote provider calls
enter as `llm_api`; ordinary tools enter as `tool_call`; network/API-oriented
tools enter as `api_request`; and `delegate_task` passes a `subagent` start
gate before child work begins. The subagent start reservation is released
before the child runs so its own tool and model requests cannot deadlock behind
their parent reservation. The cron scheduling tools (`cronjob` and
`cronjob_manage`) bypass the pool so scheduling remains untouched. Once a cron
fires, its remote model calls and ordinary tools use the same pool as
interactive work. Ollama/local-model execution remains outside this capacity
control.

For remote provider calls, admission is joint: Hermes first waits for the
provider limiter, then reserves local weighted capacity, and only then commits
the provider dispatch and starts the call. Capacity-blocked work therefore does
not consume provider dispatch accounting before it actually runs.

### Adaptive capacity

The configured `capacity_units` value is a ceiling. The resource controller
persists a learned safe capacity beneath that ceiling. Three consecutive
worsening pressure samples reduce the current limit by 20% (with a cooldown),
while eight consecutive low-pressure samples permit one small 2% recovery step.
Recovery never exceeds the configured ceiling. Adjustments, confidence,
streaks, and observation counts survive restarts in
`telemetry.resources.learned_safe_capacity.global`; reductions and increases
are also recorded as pressure and recovery events. Lowering capacity affects
only new admissions—running work is left alone.

Hermes also keeps up to 64 recurring-state profiles in
`telemetry.resources.capacity_profiles`. A profile combines coarse RAM, swap,
CPU-load, and I/O-pressure bands with a one-way fingerprint of the executable
names among the top processes (never command arguments). Once a profile has
at least three observations, returning to that state restores its learned safe
capacity before ordinary adaptive learning continues.

## Hermes integration

This uses Hermes' public plugin interfaces:

- `llm_execution` middleware to wait immediately before the real provider call.
- `tool_execution` middleware to reserve shared capacity immediately before tools run.
- `pre_api_request` to capture Hermes' own approximate input-token count.
- `post_api_request` to learn actual token usage.
- `api_request_error` to observe 429s.
- `ctx.get_config()` / `ctx.set_config()` for user-owned settings.
- `ctx.state` for profile-scoped persistent runtime state.
- `ctx.register_command()` for `/throttle` in CLI/gateway sessions such as Discord.

The plugin does **not** select providers, models, or fallbacks.

Provider/model token history informs TPM request estimates, but does not select
providers, models, fallbacks, or rate limits.

## Resource telemetry

The plugin also records lightweight host telemetry for the next stage of
resource-aware workload control. On Linux it samples memory and swap usage,
CPU utilization and load averages with short deltas/trends, memory/I/O PSI,
cgroup CPU throttling, and the top five processes by CPU/memory activity.

Persistent `telemetry.resources` storage contains bounded rolling snapshots,
resource-pressure events, recovery events, and learned safe-capacity entries.
Missing procfs, cgroup, or PSI data is reported as empty/zero rather than
failing the plugin. This telemetry and event history are observational and do
not currently change pacing.

## Install

Copy this directory to:

```bash
~/.hermes/plugins/hermes-global-throttle
```

Then:

```bash
hermes plugins enable hermes-global-throttle
hermes plugins doctor ~/.hermes/plugins/hermes-global-throttle --ci
```

Restart the Hermes process/gateway after enabling it.

## Configuration

Hermes namespaces plugin configuration under the plugin entry:

```yaml
plugins:
  entries:
    hermes-global-throttle:
      settings:
        enabled: true
        requests_per_minute: 30
        tokens_per_minute: 60000
        requests_per_day: 1000
        make_daily_quota_last_hours: 8
        longest_wait_between_requests: 15
        overrides:
          openai:
            rpm: 60
            tpm: 120000
          openai::gpt-4o:
            rpm: 120
            tpm: 200000
          anthropic:
            rpm: 50
```

The defaults above are only examples. Put in the limits for the quota you actually want Hermes to obey.

## Scoped Overrides

Overrides allow fine-grained rate-limiting at the provider level (`provider`) and model level (`provider::model`):

1. **Resolution Order**: Limits resolve hierarchically: `model override -> provider override -> global default`.
2. **Independent Pacing & FIFO**: Each configured scope maintains its own independent FIFO queue, RPM pacing, TPM ledger, in-flight reservations, and daily quota/cooldown state.
3. **Dual Satisfaction**: If both provider and model limits apply, requests must satisfy both buckets before dispatch.
4. **Non-blocking**: A blocked bucket (e.g. 429 cooldown, TPM ceiling, or RPD quota exhausted) will never block requests in unrelated buckets.

## Discord / slash commands

```text
/throttle status
/throttle on
/throttle off
/throttle rpm 30
/throttle tpm 60000
/throttle rpd 1000
/throttle hours 8
/throttle max-wait 15
/throttle overrides
/throttle get <provider|provider::model>
/throttle set <provider|provider::model> [rpm N] [tpm N] [rpd N]
/throttle remove <provider|provider::model> [rpm|tpm|rpd]
/throttle reset
/throttle reset-learning <provider> <model>
/throttle recalibrate <provider> <model>
```

The slash commands write through Hermes' plugin config and state APIs, so changes persist across restarts.

## Token pacing

Before the call, Hermes already exposes `approx_input_tokens`. The plugin
combines that with the rolling observed output average for the matching
`provider::model` to estimate the current request. If that scope has no token
history yet, it falls back to the aggregate global averages.

After a successful response, it reconciles both the scoped and aggregate
rolling token averages from the provider's real `usage` data.

This keeps TPM pacing responsive to workloads that suddenly become much larger or smaller.

## 429 behavior

The goal is to avoid 429s, not probe for them.

If a 429 nevertheless occurs:

- the learned rate factor for the impacted bucket is reduced by 20%;
- queued traffic in that bucket gets a quiet period;
- the learned penalty persists in `ctx.state`;
- every 25 successful calls lets the factor recover by 0.02, up to 1.0, only
  while that provider/model is still calibrating.

This is intentionally conservative and does not train a model or deliberately push until another 429 occurs.
Once a provider/model completes calibration with a confident safe ceiling, its
RPM and TPM values are persisted as established. Later limiter events may
reduce the active learned limits, but an established model does not
automatically recover or probe upward. Use `reset-learning` to explicitly
discard its learning, or `recalibrate` to reopen controlled calibration while
preserving history. Automatic upward recovery remains limited to models that
have not established a confident safe ceiling. Calibration advances the learned
safe rate by at most one small step per 25 successful observations, then resets
the confidence window before another upward step is considered.

`/throttle recalibrate <provider> <model>` reopens calibration for that model
without deleting its learned distributions, current safe rates, or 429 history.

## State

Persistent state includes:

- current-day request count, token count, and first-request timestamp (global and scoped);
- scoped `provider::model` rolling average total/output tokens per request,
  with aggregate global fallback averages;
- learned adaptive rate factors, cooldowns, and last 429 timestamps;
- provider and model overrides;
- Bayesian learned safe limits.

State is profile-scoped through Hermes' plugin state store.
