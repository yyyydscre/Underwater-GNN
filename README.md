# Visibility-Aware Beacon Matching

Code accompanying a study of underwater guiding-light arrays. This repository
contains the paper method only: beacon detection and center refinement,
visibility-aware graph matching, missing-beacon handling, and
identity-preserving relative pose estimation.

The underwater image dataset, annotations, trained weights, videos, and
generated experiment outputs are intentionally excluded.

## Method Components

- YOLO-based beacon detection with underwater feature-fusion extensions;
- white-core center refinement and observation reliability estimation;
- graph construction with geometry, uncertainty, and multi-scale structural
  signatures;
- visibility-aware Sinkhorn matching with an outlier/missing-beacon dustbin;
- structured partial-visibility training and temporal identity refinement;
- RANSAC-PnP, template recovery, residual-weighted refinement, and pose quality
  gating.

## Repository Layout

```text
configs/  Model, template, camera, and dataset configuration examples
src/      Detection, matching, pose, training, and evaluation code
tests/    Synthetic and unit tests that do not require the private dataset
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick Verification

The included tests use synthetic inputs and do not require private data or
trained weights.

```bash
python -m pytest -q
```

## Training and Evaluation

Pre-train the graph matcher on synthetic projected arrays:

```bash
python -m src.train_gnn_synthetic \
  --array-config configs/lamp_array_3d.yaml \
  --gnn-config configs/gnn_matcher_visibility_topology.yaml \
  --output models/gnn_matcher.pt
```

Run the complete image-sequence pipeline with user-provided detector and GNN
weights:

```bash
python -m src.run_sequence \
  --input datasets/example_sequence \
  --array-config configs/lamp_array_3d.yaml \
  --pose-array-config configs/lamp_array_pose_unity_approx.yaml \
  --detector yolo \
  --weights weights/detector.pt \
  --matcher gnn \
  --matcher-weights weights/gnn_matcher.pt \
  --output outputs/example_sequence
```

Use `python -m <module> --help` for the arguments of each training or evaluation
entry point. Dataset paths in `configs/underwater_light_box.yaml` are examples;
replace them with paths to a compatible local dataset.

## Private Data Contract

Real-data training and evaluation scripts accept user-supplied image roots and
annotation files through command-line arguments. The private dataset used in
the paper is not required to inspect the method or run the synthetic tests, and
is not included in this repository. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

## Code Availability

The permanent public repository URL will be added here and to the manuscript
after publication. See [CODE_AVAILABILITY.md](CODE_AVAILABILITY.md) for the
submission-ready statement.

## Citation

Citation metadata will be added when the manuscript bibliographic information
is finalized.
