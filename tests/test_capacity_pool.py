import threading
import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capacity_pool import DEFAULT_WORKLOAD_WEIGHTS, WeightedCapacityPool


class WeightedCapacityPoolTests(unittest.TestCase):
    def test_default_weights_cover_major_workloads(self):
        self.assertEqual(DEFAULT_WORKLOAD_WEIGHTS["tool_call"], 3.0)
        self.assertEqual(DEFAULT_WORKLOAD_WEIGHTS["subagent"], 8.0)
        self.assertEqual(DEFAULT_WORKLOAD_WEIGHTS["llm_api"], 5.0)

    def test_full_pool_delays_new_work_until_release(self):
        pool = WeightedCapacityPool(capacity=5)
        pool.acquire("first", "llm_api")
        acquired = threading.Event()

        def acquire_second():
            pool.acquire("second", "tool_call")
            acquired.set()

        thread = threading.Thread(target=acquire_second)
        thread.start()
        self.assertFalse(acquired.wait(timeout=0.05))
        pool.release("first")
        self.assertTrue(acquired.wait(timeout=1.0))
        self.assertEqual(pool.used, 3.0)
        pool.release("second")
        thread.join(timeout=1.0)
        self.assertEqual(pool.used, 0.0)

    def test_unknown_work_uses_other_weight(self):
        pool = WeightedCapacityPool(capacity=10)
        workload, weight = pool.acquire("work", "not-a-known-type")
        self.assertEqual(workload, "not-a-known-type")
        self.assertEqual(weight, DEFAULT_WORKLOAD_WEIGHTS["other"])
        self.assertTrue(pool.release("work"))
        self.assertFalse(pool.release("work"))

    def test_capacity_reduction_never_cancels_active_work(self):
        pool = WeightedCapacityPool(capacity=8)
        pool.acquire("running", "subagent")
        pool.set_capacity(4)
        self.assertIn("running", pool.active)
        self.assertEqual(pool.used, 8.0)
        self.assertTrue(pool.overcommitted)

        admitted = threading.Event()

        def acquire_new_work():
            pool.acquire("queued", "interactive")
            admitted.set()

        thread = threading.Thread(target=acquire_new_work)
        thread.start()
        self.assertFalse(admitted.wait(timeout=0.05))
        self.assertIn("running", pool.active)

        pool.release("running")
        self.assertTrue(admitted.wait(timeout=1.0))
        pool.release("queued")
        thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
