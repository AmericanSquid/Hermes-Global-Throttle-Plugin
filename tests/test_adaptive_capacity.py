import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_capacity import AdaptiveCapacityController


def snapshot(timestamp, pressure):
    return {
        "timestamp": float(timestamp),
        "memory": {"used_ratio": float(pressure)},
        "swap": {"used_ratio": 0.0},
        "cpu": {"percent": 0.0, "load1": 0.0, "count": 4},
        "io": {"psi": {}},
    }


def patterned_snapshot(timestamp, pressure, command):
    result = snapshot(timestamp, pressure)
    result["top_processes"] = [{"command": command}]
    return result


class AdaptiveCapacityControllerTests(unittest.TestCase):
    def test_reduces_after_pressure_keeps_worsening(self):
        state = {}
        controller = AdaptiveCapacityController(state)
        capacity = 100.0
        for index, score in enumerate((0.66, 0.70, 0.74, 0.78)):
            capacity = controller.observe(snapshot(index * 15, score), 100.0)

        self.assertEqual(capacity, 80.0)
        entry = state["learned_safe_capacity"]["global"]
        self.assertEqual(entry["reductions"], 1)
        self.assertEqual(entry["last_direction"], "down")
        self.assertEqual(state["pressure_events"][-1]["kind"], "capacity_reduction")

    def test_stable_system_recovers_cautiously(self):
        state = {
            "learned_safe_capacity": {
                "global": {
                    "safe_capacity": 80.0,
                    "reductions": 1,
                    "last_adjustment_at": 10.0,
                }
            }
        }
        controller = AdaptiveCapacityController(state)
        capacity = 80.0
        for index in range(8):
            capacity = controller.observe(snapshot(200 + index * 15, 0.30), 100.0)

        self.assertEqual(capacity, 81.6)
        entry = state["learned_safe_capacity"]["global"]
        self.assertEqual(entry["increases"], 1)
        self.assertEqual(entry["last_direction"], "up")
        self.assertEqual(state["recovery_events"][-1]["kind"], "capacity_increase")

    def test_instability_does_not_recover_capacity(self):
        state = {"learned_safe_capacity": {"global": {"safe_capacity": 50.0}}}
        controller = AdaptiveCapacityController(state)
        for index, score in enumerate((0.30, 0.30, 0.50, 0.30, 0.30, 0.30, 0.30, 0.30)):
            capacity = controller.observe(snapshot(200 + index * 15, score), 100.0)
        self.assertEqual(capacity, 50.0)

    def test_learned_capacity_survives_reload_and_obeys_ceiling(self):
        state = {"learned_safe_capacity": {"global": {"safe_capacity": 42.0}}}
        first = AdaptiveCapacityController(state)
        self.assertEqual(first.current_capacity(100.0), 42.0)
        reloaded = AdaptiveCapacityController(state)
        self.assertEqual(reloaded.current_capacity(100.0), 42.0)
        self.assertEqual(reloaded.current_capacity(30.0), 30.0)

    def test_recurring_resource_process_profile_restores_safe_capacity(self):
        sample = patterned_snapshot(100, 0.30, "/usr/bin/python worker.py")
        pattern = AdaptiveCapacityController.resource_pattern(sample)
        state = {
            "learned_safe_capacity": {
                "global": {"safe_capacity": 100.0, "last_pattern_key": "other"}
            },
            "capacity_profiles": {
                pattern["key"]: {
                    "safe_capacity": 42.0,
                    "observations": 4,
                    "confidence": 0.2,
                }
            },
        }

        controller = AdaptiveCapacityController(state)
        self.assertEqual(controller.observe(sample, 100.0), 42.0)
        profile = state["capacity_profiles"][pattern["key"]]
        self.assertEqual(profile["features"]["top_processes"], ["python"])
        self.assertNotIn("worker.py", str(profile["features"]))
        self.assertEqual(len(profile["features"]["process_fingerprint"]), 12)

        reloaded = AdaptiveCapacityController(state)
        self.assertEqual(reloaded.observe(sample, 100.0), 42.0)


if __name__ == "__main__":
    unittest.main()
