import sys
import unittest
from pathlib import Path

# Ensure plugin root directory is on sys.path for direct execution and discovery
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from throttle import compute_delays


class DelayMathTests(unittest.TestCase):
    def test_rpm_wins(self):
        d = compute_delays(
            rpm=30,
            tpm=1_000_000,
            estimated_tokens=100,
            remaining_requests=10_000,
            remaining_usage_seconds=0,
            max_rpd_wait=15,
        )
        self.assertAlmostEqual(d["final"], 2.0)

    def test_tpm_wins(self):
        d = compute_delays(
            rpm=120,
            tpm=60_000,
            estimated_tokens=5_000,
            remaining_requests=10_000,
            remaining_usage_seconds=0,
            max_rpd_wait=15,
        )
        self.assertAlmostEqual(d["tpm"], 5.0)
        self.assertAlmostEqual(d["final"], 5.0)

    def test_rpd_is_capped(self):
        d = compute_delays(
            rpm=30,
            tpm=60_000,
            estimated_tokens=2_000,
            remaining_requests=1000,
            remaining_usage_seconds=8 * 3600,
            max_rpd_wait=15,
        )
        self.assertAlmostEqual(d["rpd_raw"], 28.8)
        self.assertAlmostEqual(d["rpd"], 15.0)
        self.assertAlmostEqual(d["final"], 15.0)

    def test_hard_constraints_can_exceed_rpd_cap(self):
        d = compute_delays(
            rpm=30,
            tpm=60_000,
            estimated_tokens=20_000,
            remaining_requests=1000,
            remaining_usage_seconds=8 * 3600,
            max_rpd_wait=15,
        )
        self.assertAlmostEqual(d["tpm"], 20.0)
        self.assertAlmostEqual(d["final"], 20.0)

    def test_adaptive_factor_slows_rate(self):
        d = compute_delays(
            rpm=30,
            tpm=60_000,
            estimated_tokens=2_000,
            remaining_requests=10_000,
            remaining_usage_seconds=0,
            max_rpd_wait=15,
            adaptive_factor=0.8,
        )
        self.assertAlmostEqual(d["rpm"], 2.5)
        self.assertAlmostEqual(d["tpm"], 2.5)

    def test_zero_or_negative_inputs(self):
        d = compute_delays(
            rpm=0,
            tpm=-10,
            estimated_tokens=0,
            remaining_requests=0,
            remaining_usage_seconds=0,
            max_rpd_wait=15,
            adaptive_factor=0,
        )
        self.assertEqual(d["rpm"], 0.0)
        self.assertEqual(d["tpm"], 0.0)
        self.assertEqual(d["rpd_raw"], 0.0)
        self.assertEqual(d["rpd"], 0.0)
        self.assertEqual(d["final"], 0.0)


if __name__ == "__main__":
    unittest.main()

