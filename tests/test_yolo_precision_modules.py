"""Tests for the lightweight P2 precision modules."""
from __future__ import annotations

import unittest

import torch

from src.yolo_extensions import ResidualCoordinateAttention, UnderwaterCrossScaleGate


class TestYOLOPrecisionModules(unittest.TestCase):
    def test_cross_scale_gate_preserves_shape_and_learns(self) -> None:
        module = UnderwaterCrossScaleGate(32, reduction=8)
        features = torch.randn(2, 32, 12, 12, requires_grad=True)
        output = module(features)
        self.assertEqual(output.shape, features.shape)
        output.square().mean().backward()
        self.assertIsNotNone(module.selector[0].weight.grad)
        self.assertGreater(float(module.selector[0].weight.grad.abs().sum()), 0.0)

    def test_coordinate_attention_preserves_shape_and_learns(self) -> None:
        module = ResidualCoordinateAttention(16, reduction=8)
        features = torch.randn(2, 16, 10, 14, requires_grad=True)
        output = module(features)
        self.assertEqual(output.shape, features.shape)
        output.square().mean().backward()
        self.assertGreater(float(module.height_gate.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.width_gate.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
