"""CropGuard - edge AI pest and disease detection for the Smart Farming Assistant.

The package is deliberately layered so that the heavy training stack (PyTorch)
never has to be installed on a field device:

``cropguard.taxonomy``      class registry + agronomic metadata (pure python)
``cropguard.advisory``      class -> farmer-facing recommendation (pure python)
``cropguard.early_warning`` temporal pest pressure + weather disease risk (pure python)
``cropguard.edge``          numpy/onnxruntime inference runtime (no torch needed)
``cropguard.data``          datasets, augmentation, synthetic data (torch)
``cropguard.models``        backbones and heads (torch)
``cropguard.train`` / ``evaluate`` / ``export`` / ``benchmark``  (torch)
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
