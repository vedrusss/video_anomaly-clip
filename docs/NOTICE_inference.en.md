# NOTICE: Inference

## Summary

Inference in this project is a three-stage pipeline:

1. `preprocess` - read the video, sample frames, apply resize/crop/normalization.
2. `encoding` - run the selected frames through the CLIP visual encoder (`ViT-B/16`) and build feature vectors.
3. `classify` - run those feature vectors through `AnomalyCLIP` using the loaded checkpoint.

Here `CLIP visual encoder ViT-B/16` means the visual branch of OpenAI CLIP. It turns each frame into a compact embedding vector, and those embeddings are then consumed by the upper AnomalyCLIP logic.

## Key constraints

- The default inference setup is tightly coupled to `num_segments = 32` and `seg_length = 16`.
- That gives a base input length of `32 * 16 = 512`.
- Without retraining, or at least a non-trivial architectural adaptation, this shape should be treated as fixed.
- In the current code, these values are hardcoded in `inference/common.py`, not exposed as CLI arguments.

In other words, `512` is not just a tensor size. It is the structural sequence length that reshape operations, the temporal module, and part of the loss logic are built around.

## Timing breakdown

On your run, the rough picture was:

- `preprocess` - about `1.27s`
- `encoding` - about `3.43s`
- `classify` - only a few milliseconds, around `5.5 ms` in your measurements

This was for video `G4GVK6ZQ-336-30005216_4N72787A6CFA7_1514327102059.mp4` and checkpoint `checkpoints/shanghaitech/last.ckpt`.

Practical takeaway:

- The CLIP encoder is the main cost in inference.
- The `classify` stage is relatively light because it works on ready-made feature vectors.
- Padding up to 512 makes even the repeated tail frames expensive to process.

The practical meaning is that the current bottleneck is not the top-level classifier, but the repeated pass through the CLIP visual encoder.

## Frame sampling

- Raw videos are sampled with `fps`, defaulting to `fps=8`.
- This is not a key-frame detector, just a uniform downsampling policy.
- For videos with sparse events, this is not the most informative strategy.

`fps=8` means the script keeps roughly 8 frames per second if the source video has a higher FPS. This is a coarse sampling heuristic, not a semantic frame selector.

### Practical drawback

If the input still has to be expanded to 512, a better strategy would be:

- pick the most informative frames first;
- then pad to the required multiple.

The current approach is simpler:

- sample by FPS;
- pad by repeating the last frame until the tensor reaches the required length.

## Padding

Padding is currently implemented by repeating the last frame.

That is shape-correct, but not compute-efficient:

- the CLIP encoder still processes those duplicates;
- for short videos, padded duplicates can dominate the encoding cost.

So part of the compute can go to repeated copies of the final frame rather than genuinely new content.

### More efficient idea

Without changing model logic, you could:

- encode the last frame once;
- then pad feature vectors instead of re-encoding the same image.

That would preserve behavior while reducing wasted CLIP work.

The idea is simple: do `image -> CLIP` once for the tail frame, then repeat the resulting feature vector instead of re-running the same image through CLIP many times.

## `ncentroid` note

- If `ncentroid.pt` is available next to the checkpoint, inference will use it.
- Otherwise, it can compute a centroid from the current video on the fly.
- For new videos this is supported, but a saved centroid is usually more stable.

`ncentroid` is the mean vector of the normal feature space, i.e. an averaged embedding of normal frames/features. It is used to center image/text features before matching them.

Practically, inference does not break if a dedicated `ncentroid.pt` is missing, but a centroid saved during training usually keeps behavior closer to the original train setup.
