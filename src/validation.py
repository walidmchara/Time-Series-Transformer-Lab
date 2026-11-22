from __future__ import annotations


def validate_feature_columns(features, target):
    if not isinstance(target, str) or not target.strip():
        raise ValueError("The target column must be a non-empty string")
    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError("Feature columns must be a non-empty list")
    if any(not isinstance(name, str) or not name.strip() for name in features):
        raise ValueError("Feature names must be non-empty strings")
    if target in features:
        raise ValueError("Target column cannot also be an input feature")
    if len(set(features)) != len(features):
        raise ValueError("Feature columns must not contain duplicates")
