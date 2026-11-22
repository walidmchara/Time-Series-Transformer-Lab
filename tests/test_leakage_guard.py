import unittest

from src.validation import validate_feature_columns


class FeatureLeakageTests(unittest.TestCase):
    def test_distinct_features_are_accepted(self):
        validate_feature_columns(["temperature", "wind"], "power")

    def test_target_in_features_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Target column"):
            validate_feature_columns(["temperature", "power"], "power")

    def test_duplicate_or_empty_features_are_rejected(self):
        for features in ([], ["temperature", "temperature"]):
            with self.subTest(features=features), self.assertRaises(ValueError):
                validate_feature_columns(features, "power")


if __name__ == "__main__":
    unittest.main()
