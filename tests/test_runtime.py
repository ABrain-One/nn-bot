import unittest

import numpy as np

from types import SimpleNamespace

from ab.bot.runtime import (
    device_name,
    message_stamp_ns,
    preprocess_image,
    synthetic_rgb_frame,
    timing_stats,
)


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
            {
                "avg": 20.0,
                "median": 20.0,
                "p95": 29.0,
                "min": 10.0,
                "max": 30.0,
                "std": 10.0,
                "samples": [10.0, 20.0, 30.0],
            },
        )

    def test_timing_statistics_handle_one_sample(self):
        self.assertEqual(
            timing_stats([10]),
            {
                "avg": 10.0,
                "median": 10.0,
                "p95": 10.0,
                "min": 10.0,
                "max": 10.0,
                "std": 0.0,
                "samples": [10.0],
            },
        )

    def test_timing_statistics_keep_samples_and_report_percentiles(self):
        stats = timing_stats(list(range(1, 21)))

        self.assertEqual(len(stats["samples"]), 20)
        self.assertEqual(stats["median"], 10.5)
        self.assertAlmostEqual(stats["p95"], 19.05)

    def test_device_name_is_not_empty(self):
        self.assertTrue(device_name())

    def test_synthetic_rgb_frame_has_camera_dimensions_and_changes(self):
        first = synthetic_rgb_frame(64, 48)
        next_frame = synthetic_rgb_frame(64, 48, frame_index=1)

        self.assertEqual(first.shape, (48, 64, 3))
        self.assertEqual(first.dtype, np.uint8)
        self.assertEqual(first[0, -1, 0], 255)
        self.assertEqual(first[-1, 0, 1], 255)
        self.assertFalse(np.array_equal(first, next_frame))

    def test_synthetic_rgb_frame_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            synthetic_rgb_frame(0, 48)

    def test_message_stamp_is_reported_in_nanoseconds(self):
        message = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=34))
        )

        self.assertEqual(message_stamp_ns(message), 12_000_000_034)


if __name__ == "__main__":
    unittest.main()