import unittest

import numpy as np
import torch

from src.graph import GuidingLightGNN, knn_edges, observation_features
from src.schema import LightDetection
from src.compare_gnn_fewshot import _transport_loss
from src.test_gnn_real_sequence_holdout import (
    augment_geometry_v2,
    augment_structured_occlusion,
    summarize_predictions,
)
from src.config import load_array_configuration
from src.matcher import LampArrayMatcher


def make_detection(index: int) -> LightDetection:
    return LightDetection(
        xy=np.asarray([20.0 + 12.0 * index, 30.0 + 7.0 * index]),
        confidence=0.75,
        color_probs=np.asarray([0.8, 0.1, 0.1], dtype=np.float32),
        color="green",
        radius=5.0,
        brightness=180.0,
        source="unit_test",
        center_quality=0.8,
        detector_confidence=0.76,
        center_confidence=0.82,
        center_covariance=np.asarray([[0.25, 0.04], [0.04, 0.36]]),
        center_valid=True,
    )


class GNNUncertaintyTest(unittest.TestCase):
    def test_too_few_observations_are_explicitly_rejected(self):
        configuration = load_array_configuration("configs/lamp_array_3d.yaml")
        matcher = LampArrayMatcher(configuration, mode="geometry")
        result = matcher._match_gnn(
            [make_detection(index) for index in range(3)]
        )
        self.assertEqual(result.method, "gnn_insufficient_observations")
        self.assertEqual(result.missing_lamp_ids, configuration.ids)
        self.assertFalse(result.matches)

    def test_uncertainty_features_are_finite_and_keep_input_width(self):
        for feature_mode in (
            "geometry_uncertainty",
            "geometry_uncertainty_relative",
        ):
            features = observation_features(
                [make_detection(index) for index in range(5)],
                color_mode="neutral",
                feature_mode=feature_mode,
            )
            self.assertEqual(features.shape, (5, 10))
            self.assertTrue(np.all(np.isfinite(features)))
            self.assertTrue(np.all(features[:, 6:8] >= 0.0))

    def test_gated_model_supports_gradient_updates(self):
        detections = [make_detection(index) for index in range(5)]
        points = np.asarray([item.xy for item in detections], dtype=np.float32)
        edges, edge_attrs = knn_edges(points, 3)
        model = GuidingLightGNN(
            hidden_dim=32,
            layers=2,
            cross_layers=1,
            heads=4,
            edge_gating=True,
        )
        features = torch.from_numpy(
            observation_features(
                detections,
                feature_mode="geometry_uncertainty",
            )
        )
        output = model(
            features,
            torch.from_numpy(edges),
            torch.from_numpy(edge_attrs),
            features,
            torch.from_numpy(edges),
            torch.from_numpy(edge_attrs),
        )
        output.sum().backward()
        gate_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if ".gate." in name
        ]
        self.assertTrue(gate_gradients)
        self.assertTrue(all(gradient is not None for gradient in gate_gradients))

    def test_global_context_and_symmetric_pair_scoring_backpropagate(self):
        detections = [make_detection(index) for index in range(5)]
        points = np.asarray([item.xy for item in detections], dtype=np.float32)
        edges, edge_attrs = knn_edges(points, 3)
        model = GuidingLightGNN(
            hidden_dim=32,
            layers=2,
            cross_layers=1,
            heads=4,
            global_context_layers=1,
            symmetric_pair_scoring=True,
            cosine_similarity=True,
        )
        features = torch.from_numpy(
            observation_features(
                detections,
                feature_mode="geometry_uncertainty",
            )
        )
        output = model(
            features,
            torch.from_numpy(edges),
            torch.from_numpy(edge_attrs),
            features,
            torch.from_numpy(edges),
            torch.from_numpy(edge_attrs),
        )
        self.assertEqual(tuple(output.shape), (5, 5))
        output.square().mean().backward()
        self.assertIsNotNone(model.log_similarity_scale.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.global_context_layers.parameters()
            )
        )

    def test_visibility_head_adjusts_scores_and_backpropagates(self):
        detections = [make_detection(index) for index in range(5)]
        points = np.asarray([item.xy for item in detections], dtype=np.float32)
        edges, edge_attrs = knn_edges(points, 3)
        model = GuidingLightGNN(
            hidden_dim=32,
            layers=2,
            cross_layers=1,
            heads=4,
            visibility_aware=True,
            visibility_score_weight=0.25,
        )
        features = torch.from_numpy(observation_features(detections))
        scores, missing_logits = model.forward_with_aux(
            features,
            torch.from_numpy(edges),
            torch.from_numpy(edge_attrs),
            features,
            torch.from_numpy(edges),
            torch.from_numpy(edge_attrs),
        )
        self.assertEqual(tuple(scores.shape), (5, 5))
        self.assertEqual(tuple(missing_logits.shape), (5,))
        (scores.mean() + missing_logits.mean()).backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in model.template_missing_head.parameters()
            )
        )

    def test_visibility_loss_prefers_correct_missing_template(self):
        transport = torch.full((4, 5), -2.0)
        labels = [0, 1, 2]
        correct_missing = torch.tensor([-4.0, -4.0, -4.0, 4.0])
        wrong_missing = -correct_missing
        correct_loss = _transport_loss(
            transport,
            labels,
            4,
            missing_template_weight=0.0,
            missing_logits=correct_missing,
            visibility_loss_weight=1.0,
        )
        wrong_loss = _transport_loss(
            transport,
            labels,
            4,
            missing_template_weight=0.0,
            missing_logits=wrong_missing,
            visibility_loss_weight=1.0,
        )
        self.assertLess(float(correct_loss), float(wrong_loss))

    def test_docking_v2_keeps_detection_label_alignment(self):
        detections = [make_detection(index) for index in range(10)]
        labels = list(range(10))
        augmented, augmented_labels = augment_geometry_v2(
            detections,
            labels,
            np.random.default_rng(42),
        )
        self.assertEqual(len(augmented), len(augmented_labels))
        self.assertGreaterEqual(len(augmented), 4)
        positive_labels = [label for label in augmented_labels if label >= 0]
        self.assertEqual(len(positive_labels), len(set(positive_labels)))
        self.assertTrue(all(0 <= label < 10 for label in positive_labels))

    def test_structured_occlusion_masks_only_existing_lamps(self):
        detections = [make_detection(index) for index in range(13)]
        labels = list(range(13))
        augmented, augmented_labels = augment_structured_occlusion(
            detections,
            labels,
            np.random.default_rng(4),
            probability=1.0,
            maximum_drop=3,
            minimum_visible=7,
        )
        self.assertLess(len(augmented), len(detections))
        self.assertEqual(len(augmented), len(augmented_labels))
        self.assertTrue(set(augmented_labels).issubset(set(labels)))

    def test_summary_reports_missing_id_and_visibility_metrics(self):
        configuration = load_array_configuration("configs/lamp_array_3d.yaml")
        records = [
            {
                "frame_name": "partial.jpg",
                "labels": list(range(10)),
                "predictions": list(range(10)),
            }
        ]
        metrics = summarize_predictions(records, configuration)
        self.assertEqual(metrics["missing_id_f1"], 1.0)
        self.assertEqual(
            metrics["id_accuracy_by_visibility"]["partial_10_12"]["id_accuracy"],
            1.0,
        )
        self.assertEqual(metrics["four_correct_support_rate"], 1.0)
        self.assertEqual(metrics["mean_correct_ids_per_frame"], 10.0)

    def test_layer_auxiliary_loss_penalizes_layer_swaps(self):
        correct = torch.full((3, 14), -8.0)
        swapped = torch.full((3, 14), -8.0)
        labels = [0, 2, 8]
        for row, label in enumerate(labels):
            correct[row, label] = -0.1
            swapped_label = label + 7 if label < 7 else label - 7
            swapped[row, swapped_label] = -0.1
        correct = torch.cat(
            [correct, torch.full((1, 14), -8.0)],
            dim=0,
        )
        swapped = torch.cat(
            [swapped, torch.full((1, 14), -8.0)],
            dim=0,
        )
        correct_loss = _transport_loss(
            correct,
            labels,
            13,
            layer_loss_weight=0.25,
        )
        swapped_loss = _transport_loss(
            swapped,
            labels,
            13,
            layer_loss_weight=0.25,
        )
        self.assertLess(float(correct_loss), float(swapped_loss))

    def test_hard_negative_loss_penalizes_small_same_layer_margin(self):
        separated = torch.full((2, 14), -8.0)
        ambiguous = torch.full((2, 14), -8.0)
        separated[0, 4] = -0.1
        separated[0, 5] = -5.0
        ambiguous[0, 4] = -0.1
        ambiguous[0, 5] = -0.2
        separated[1] = -8.0
        ambiguous[1] = -8.0
        separated_loss = _transport_loss(
            separated,
            [4],
            13,
            hard_negative_weight=0.1,
            hard_negative_margin=2.0,
        )
        ambiguous_loss = _transport_loss(
            ambiguous,
            [4],
            13,
            hard_negative_weight=0.1,
            hard_negative_margin=2.0,
        )
        self.assertLess(float(separated_loss), float(ambiguous_loss))

    def test_frame_hard_loss_emphasizes_the_weakest_assignment(self):
        transport = torch.full((4, 4), -6.0)
        transport[0, 0] = -0.1
        transport[1, 1] = -0.2
        transport[2, 2] = -3.0
        base = _transport_loss(
            transport,
            [0, 1, 2],
            3,
            missing_template_weight=0.0,
        )
        structured = _transport_loss(
            transport,
            [0, 1, 2],
            3,
            missing_template_weight=0.0,
            frame_hard_weight=0.15,
            frame_hard_temperature=0.5,
        )
        self.assertGreater(float(structured), float(base))


if __name__ == "__main__":
    unittest.main()
