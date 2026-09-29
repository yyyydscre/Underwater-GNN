# Data Availability

The dataset used in the manuscript is not publicly distributed.

This code repository intentionally excludes:

- underwater source images and videos;
- YOLO detection and keypoint labels;
- frame-level beacon identity annotations;
- dataset archives and train/validation/test split files containing private paths;
- trained detector and graph-matching checkpoints;
- generated experiment outputs.

The dataset configuration files in `configs/` contain local placeholder paths only. Users must provide their own data and calibration parameters to train or evaluate the method.
