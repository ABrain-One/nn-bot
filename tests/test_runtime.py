import unittest

import numpy as np

from ab.bot.runtime import device_name, preprocess_image, timing_stats


class RuntimeTests(unittest.TestCase):
    def test_preprocesses_nchw_input_and_resizes(self):
        image = np.full((4, 6, 3), 255, dtype=np.uint8)

        result = preprocess_image(image, [1, 3, 2, 3])

        self.assertEqual(result.shape, (1, 3, 2, 3))
        np.testing.assert_array_equal(result, np.ones((1, 3, 2, 3), dtype=np.float32))

    def test_preprocesses_nhwc_input_with_normalization(self):
        image = np.full((2, 2, 3), 128, dtype=np.uint8)

        result = preprocess_image(image, [1, 2, 2, 3], scale=1.0, mean=[1, 2, 3])

        self.assertEqual(result.shape, (1, 2, 2, 3))
        np.testing.assert_array_equal(result[0, 0, 0], [127, 126, 125])

    def test_rejects_ambiguous_model_input_shape(self):
        with self.assertRaises(ValueError):
            preprocess_image(np.zeros((2, 2, 3)), [1, 7, 8, 9])

    def test_timing_statistics_use_nanoseconds(self):
        self.assertEqual(
            timing_stats([10, 20, 30]),
            {"avg": 20.0, "min": 10.0, "max": 30.0, "std": 10.0},
        )

    def test_timing_statistics_handle_one_sample(self):
        self.assertEqual(
            timing_stats([10]),
            {"avg": 10.0, "min": 10.0, "max": 10.0, "std": 0.0},
        )

        def test_device_name_is_not_empty(self):
                self.assertTrue(device_name())


if __name__ == "__main__":
    unittest.main()