# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

openWakeWord is an audio wake word / phrase detection framework. This is the **skoky fork**, whose recent work centers on the *custom model training pipeline* (synthetic data generation via Piper / ElevenLabs TTS, metadata + config export alongside trained models, language handling). The upstream library and pre-trained models are otherwise intact.

## Commands

```bash
# Install for inference only
pip install -e .

# Install with full training/dev dependencies (torch, tensorflow-cpu, onnx, audiomentations, etc.)
pip install -e .[full]      # training
pip install -e .[test]      # lint + type-check + test deps only

# Run the test suite (pytest config in pyproject.toml auto-adds coverage + flake8 + mypy)
pytest

# Run a single test
pytest tests/test_models.py::TestModels::test_models -v

# Lint / type-check on their own (same checks pytest runs)
flake8 openwakeword tests          # max line length 140
mypy openwakeword --ignore-missing-imports
```

CI (`.github/workflows/`) runs `pytest` on Python 3.8. `pytest` collects from both `tests/` and `openwakeword/` and enforces flake8 + mypy as part of the run, so a lint/type failure fails the test suite.

## Inference architecture

The core insight: all wake word models share a **single audio preprocessing front-end**, and each wake word is a small classifier on top of shared embeddings.

1. **Shared front-end** (`AudioFeatures` in `openwakeword/utils.py`): raw 16-bit 16 kHz PCM → melspectrogram model → Google `speech_embedding` model → feature vectors. These two feature models live in `resources/models/` and are downloaded via `openwakeword.utils.download_models()`.
2. **Per-wakeword classifiers** (`Model` in `openwakeword/model.py`): each loaded `.tflite`/`.onnx` model consumes the shared embeddings and outputs a score. `Model.predict(frame)` streams audio frame-by-frame (maintains internal buffers); `Model.predict_clip(path)` runs a whole WAV file.
3. **Inference frameworks**: every model exists in both `.tflite` and `.onnx`. `inference_framework="tflite"` is the default (better on x86/ARM64); `"onnx"` is the alternative. Code paths branch on this string throughout `model.py` and `utils.py` — keep both in sync when touching prediction logic.
4. **Optional filters**: Silero VAD (`vad.py`, gated by `vad_threshold`), Speex noise suppression, and per-user **custom verifier models** (`custom_verifier_model.py`) that re-score frames above a threshold to cut false accepts.

The pre-trained model registry (`MODELS`, `FEATURE_MODELS`, `VAD_MODELS`) and their download URLs are defined in `openwakeword/__init__.py`.

## Training architecture

Training is driven by `openwakeword/train.py` as a CLI against a YAML config (see `examples/custom_model.yml`, which is the authoritative documentation of every config key):

```bash
python openwakeword/train.py --training_config my_model.yml --generate_clips --augment_clips --train_model
```

The flags are independent pipeline stages, normally run in order:
- `--generate_clips`: synthesize positive + adversarial-negative TTS clips. Imports `generate_samples` from a **separate `piper-sample-generator` repo** whose path is given by `piper_sample_generator_path` in the config (cloned into `notebooks/piper-sample-generator/`).
- `--augment_clips`: mix clips with room impulse responses (`rir_paths`) and background audio (`background_paths`), then precompute openWakeWord features to `.npy`. `--overwrite` forces recomputation.
- `--train_model`: train the classifier and **export ONNX + tflite plus `<model>.metadata.json` and `<model>.config.json`** (the metadata/config export is a fork-specific addition — see `Model.export_model` in `train.py`).

Training data plumbing lives in `openwakeword/data.py`: synthetic generation, `augment_clips`, `mix_clips_batch`, reverb/augmentation, adversarial-text generation, and `mmap_batch_generator` — a memory-mapped batch loader so feature `.npy` files can exceed RAM (a fast NVMe is assumed). The training model definition and loop (`Model` nn.Module, `auto_train`, `train_model`) are in `train.py` — note this is a **different `Model` class** than the inference one in `model.py`.

## Notebooks

`notebooks/automatic_model_training.ipynb` is the recommended end-to-end training path; `notebooks/training_models.ipynb` is the educational walkthrough (not for production models). `notebooks/openwakeword/` and `notebooks/piper-sample-generator/` are nested clones (untracked) used by the notebooks.

## Conventions

- Audio is always 16-bit, 16 kHz, mono PCM. Frames should be multiples of 80 ms (1280 samples) for efficiency.
- When adding a model format or prediction path, update **both** tflite and onnx branches.
- flake8 max line length is **140**, not the default 79.
