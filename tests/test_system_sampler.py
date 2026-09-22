import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from system_sampler import SystemResourceSampler


class SystemResourceSamplerTests(unittest.TestCase):
    def test_snapshot_shape_and_bounded_history(self):
        state = {}
        sampler = SystemResourceSampler(state, interval_seconds=60.0)

        first = sampler.maybe_sample(force=True)
        self.assertIsNotNone(first)
        self.assertIn("memory", first)
        self.assertIn("swap", first)
        self.assertIn("cpu", first)
        self.assertIn("io", first)
        self.assertIn("throttling", first)
        self.assertIn("top_processes", first)
        self.assertEqual(len(state["history"]), 1)

        # Forced samples are useful for diagnostics, but history stays bounded.
        for _ in range(125):
            sampler.maybe_sample(force=True)
        self.assertEqual(len(state["history"]), 120)
        self.assertIs(state["latest"], state["history"][-1])

    def test_interval_skips_duplicate_samples(self):
        state = {}
        sampler = SystemResourceSampler(state, interval_seconds=60.0)
        self.assertIsNotNone(sampler.maybe_sample(force=True))
        self.assertIsNone(sampler.maybe_sample())
        self.assertEqual(len(state["history"]), 1)

    def test_pressure_recovery_and_capacity_storage(self):
        state = {}
        sampler = SystemResourceSampler(state)
        pressured = {
            "timestamp": 100.0,
            "memory": {"used_ratio": 0.95},
            "swap": {"total_bytes": 0, "used_ratio": 0.0},
            "cpu": {"count": 4, "load1": 8.0},
            "io": {"psi": {}},
            "throttling": {"cgroup_cpu": {"nr_throttled": 2}},
        }
        sampler._update_pressure_events(pressured)
        self.assertEqual({"memory", "cpu", "throttling"}, {
            event["kind"] for event in state["pressure_events"]
        })

        recovered = {
            "timestamp": 110.0,
            "memory": {"used_ratio": 0.50},
            "swap": {"total_bytes": 0, "used_ratio": 0.0},
            "cpu": {"count": 4, "load1": 0.5},
            "io": {"psi": {}},
            "throttling": {"cgroup_cpu": {"nr_throttled": 2}},
        }
        sampler._update_pressure_events(recovered, pressured)
        self.assertEqual({"memory", "cpu", "throttling"}, {
            event["kind"] for event in state["recovery_events"]
        })

        entry = sampler.record_learned_safe_capacity(
            "normal::interactive", 4.0, confidence=0.8, observations=12, timestamp=120.0
        )
        self.assertEqual(entry["safe_capacity"], 4.0)
        self.assertEqual(state["learned_safe_capacity"]["normal::interactive"]["observations"], 12)

        reloaded = SystemResourceSampler(state)
        self.assertIn("normal::interactive", reloaded._state["learned_safe_capacity"])


if __name__ == "__main__":
    unittest.main()
