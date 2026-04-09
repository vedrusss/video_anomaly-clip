# NOTICE: Training

## Summary

Training in this project is **transfer learning**, not full end-to-end CLIP training.

The default pipeline is feature-based:

- the dataset reads precomputed `.npy` features;
- the CLIP visual encoder is frozen;
- only the upper task-specific parts are trained.

This means the default setup is not end-to-end training from raw video frames to the final prediction. Instead, only the task-specific head on top of CLIP features is updated.

## What is actually trained

The training module updates:

- `PromptLearner`
- `SelectorModel`
- `TemporalModel`
- `text_projection` in the CLIP text branch

`PromptLearner` learns the prompt context, `SelectorModel` aligns image/text features and selects relevant snippets, and `TemporalModel` performs temporal aggregation.

Frozen parts include:

- `image_encoder`
- the main CLIP text transformer
- `token_embedding`
- text `positional_embedding`
- text `ln_final`

That is what keeps training comparatively light: the large CLIP backbone is not updated.

## What the training pipeline consumes

By default, training does not use raw video frames:

- the data config sets `load_from_features: True`;
- the datamodule selects the feature-based dataset;
- that dataset reads `.npy` feature arrays instead of images.

In practice, this means the video has already been encoded once and saved to disk, and training then operates only on those stored feature vectors.

This matters because:

- the visual encoder is frozen, so re-running it every iteration would be wasted work;
- precomputed features make training much more efficient.

So your intuition is correct: if the encoder is not trainable, repeatedly recomputing it during training would just waste time.

## `ncentroid`

Training computes and stores `ncentroid`:

- it is the mean vector of normal features;
- it is used to center image/text features;
- it is saved as `ncentroid.pt`.

This is not model weight. It is a helper statistic that normalizes the feature space around the normal class.

During training it is computed either:

- from the feature-based train loader,
- or via `image_encoder` if raw-video mode is enabled.

So the centroid can come from precomputed features or from direct frame encoding, but the feature-based mode is the natural default.

## What changes for new classes

If you add your own anomaly classes or replace existing ones:

- update `labels_file`;
- update `num_classes`;
- check `normal_id`;
- keep data/model/loss configs consistent;
- recompute `ncentroid`;
- fine-tune the trainable parts.

When you change the class list, remember that class names are used not only for reporting but also for building the text prompts in the CLIP text branch.

### If the number of classes stays the same

If you replace some classes but keep the same `num_classes`:

- you can usually start from an existing checkpoint;
- keep `image_encoder` frozen;
- reinitialize and fine-tune `PromptLearner`;
- fine-tune `SelectorModel`, `TemporalModel`, and `text_projection`.

This is the mildest case: the output shape stays the same, but the semantics of the classes change, so old weights are useful as an initialization, not as the final solution.

### If the number of classes changes

If you add new classes and change `num_classes`:

- the old checkpoint is no longer fully compatible;
- some layers will need reinitialization;
- fine-tuning on the new class set is required.

In that case, `labels_file`, `normal_id`, and all places where class count affects tensor shapes need to be aligned again.

## Practical takeaway

- Full from-scratch training is usually not necessary.
- The minimal sensible path is fine-tuning from an existing checkpoint.
- But once class layout or input shape changes, this is no longer just a new inference run - it becomes a new training setup.

In short, when you introduce new classes, the right question is not "reuse the model or not", but "which parts of the old checkpoint can be preserved and which parts need retraining".
