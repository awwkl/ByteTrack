# Evaluation of frame-skipping and interpolation using ByteTrack

## Attribution
This code is based on the original ByteTrack source code [(GitHub)](https://github.com/ifzhang/ByteTrack)

## Overview
- On top of the original ByteTrack, this repository adds evaluation of performance when frames are skipped and interpolation is used
- Most of the additional code is located in `awwkl_dir/benchmark_skipping/` and `tools/demo_track.py`
- Everything in this repository is available to the public. Nothing is sensitive.

## Instructions

### If you wish to view the results
- Look into `awwkl_dir/benchmark_skipping/results_folder/`
- You will see the evaluation results for:
   - interpolating detections vs tracks
   - varying # of frames skipped
- E.g. `awwkl_dir/benchmark_skipping/results_folder/dets_skip_2/` means:
   - Interpolate detections
   - 2 frames were skipped (i.e. every 1 of 3 frames were passed to the object detector)

### If you wish to run the evaluation
- Follow the ByteTrack repo:
   - Set up the MOT17 datasets
   - Set up the python requirements
- Look through `awwkl_dir/benchmark_skipping/` for how to setup and run the evaluation. Non-exhaustive guideline of the steps:
   - Create the videos from the image frames in MOT17 datasets
   - Create the ground truth text files
   - Run `tools/demo_track.py` to generate the MOT outputs with frame-skipping
   - Run `awwkl_dir/benchmark_skipping/eval.py` for evaluation