import math
import unittest

from kv_cache import KnormPress, MultiHeadAttentionWithKVCache


class KVCacheAttentionTests(unittest.TestCase):
    def test_cache_grows_and_reuses_previous_values(self):
        mha = MultiHeadAttentionWithKVCache(num_heads=1, head_dim=2)

        out1, _ = mha.forward(q=[[1.0, 0.0]], k=[[1.0, 0.0]], v=[[2.0, 0.0]])
        self.assertEqual(mha.cache_size, 1)
        self.assertAlmostEqual(out1[0][0], 2.0, places=6)
        self.assertAlmostEqual(out1[0][1], 0.0, places=6)

        out2, _ = mha.forward(q=[[1.0, 0.0]], k=[[0.0, 1.0]], v=[[0.0, 4.0]])
        self.assertEqual(mha.cache_size, 2)

        score_prev = (1.0 / math.sqrt(2.0))
        score_new = 0.0
        w_prev = math.exp(score_prev) / (math.exp(score_prev) + math.exp(score_new))
        w_new = 1.0 - w_prev
        self.assertAlmostEqual(out2[0][0], 2.0 * w_prev, places=6)
        self.assertAlmostEqual(out2[0][1], 4.0 * w_new, places=6)

    def test_reset_cache(self):
        mha = MultiHeadAttentionWithKVCache(num_heads=2, head_dim=2)
        mha.forward(
            q=[[1.0, 0.0], [0.0, 1.0]],
            k=[[1.0, 0.0], [0.0, 1.0]],
            v=[[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertEqual(mha.cache_size, 1)
        mha.reset_cache()
        self.assertEqual(mha.cache_size, 0)


class KnormPressTests(unittest.TestCase):
    def test_knormpress_keeps_high_norm_tokens(self):
        cache_k = [[
            [1.0, 0.0],   # norm 1
            [3.0, 0.0],   # norm 3
            [0.0, 2.0],   # norm 2
            [4.0, 0.0],   # norm 4
        ]]
        cache_v = [[
            [10.0, 0.0],
            [30.0, 0.0],
            [0.0, 20.0],
            [40.0, 0.0],
        ]]

        compressor = KnormPress(max_cache_tokens=2)
        stats = compressor.compress(cache_k, cache_v)

        self.assertEqual(cache_k[0], [[3.0, 0.0], [4.0, 0.0]])
        self.assertEqual(cache_v[0], [[30.0, 0.0], [40.0, 0.0]])
        self.assertEqual(stats.kept_indices_by_head[0], [1, 3])
        self.assertEqual(stats.removed_indices_by_head[0], [0, 2])

    def test_forward_with_compression_caps_cache_size(self):
        mha = MultiHeadAttentionWithKVCache(num_heads=1, head_dim=2)
        compressor = KnormPress(max_cache_tokens=1)

        mha.forward(q=[[1.0, 0.0]], k=[[1.0, 0.0]], v=[[1.0, 0.0]], compressor=compressor)
        self.assertEqual(mha.cache_size, 1)

        _, stats = mha.forward(q=[[1.0, 0.0]], k=[[0.5, 0.5]], v=[[0.0, 1.0]], compressor=compressor)
        self.assertEqual(mha.cache_size, 1)
        self.assertEqual(stats.removed_indices_by_head[0], [1])


if __name__ == "__main__":
    unittest.main()
