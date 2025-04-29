"""Temporary pass-through until we migrate code out of legacy."""
from .legacy.feature_engineering import FeatureEngine as _L
add_advanced_features = _L.integrate_advanced_features
