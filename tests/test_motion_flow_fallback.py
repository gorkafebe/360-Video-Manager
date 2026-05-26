import unittest

from detector.motion_analysis import build_flow_fallback_chain


class MotionFlowFallbackChainTests(unittest.TestCase):
    def test_robust_chain_keeps_preferred_then_compatible_then_fallback(self):
        capabilities = {
            "has_dis": True,
            "has_tvl1": True,
            "has_deepflow": True,
            "has_pcaflow": True,
            "has_sparse_to_dense": True,
        }
        chain = build_flow_fallback_chain(
            preferred_algorithm="deepflow",
            capabilities=capabilities,
            profile="robust",
        )
        self.assertEqual(chain[0], "deepflow")
        self.assertEqual(chain[1], "tvl1")
        self.assertEqual(chain[2], "dis")
        self.assertEqual(chain[-1], "farneback")

    def test_robust_chain_falls_back_when_preferred_unavailable(self):
        capabilities = {
            "has_dis": False,
            "has_tvl1": True,
            "has_deepflow": False,
            "has_pcaflow": True,
            "has_sparse_to_dense": False,
        }
        chain = build_flow_fallback_chain(
            preferred_algorithm="deepflow",
            capabilities=capabilities,
            profile="robust",
        )
        self.assertEqual(chain[0], "tvl1")
        self.assertIn("pcaflow", chain)
        self.assertEqual(chain[-1], "farneback")

    def test_baseline_chain_keeps_farneback_safe_default(self):
        capabilities = {
            "has_dis": False,
            "has_tvl1": True,
            "has_deepflow": True,
            "has_pcaflow": True,
            "has_sparse_to_dense": True,
        }
        chain = build_flow_fallback_chain(
            preferred_algorithm="farneback",
            capabilities=capabilities,
            profile="baseline",
        )
        self.assertEqual(chain[0], "farneback")
        self.assertEqual(chain[-1], "farneback")


if __name__ == "__main__":
    unittest.main()
