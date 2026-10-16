# v0.1.0 verification record

Date: 2026-10-16. Primary development machine: Apple M3, 16 GB unified memory, macOS 26.5.2. Python 3.12.14 for the app; existing staged Qwen environment uses Python 3.11. FFmpeg 9.0.2 is the locally tested renderer.

## What the checks mean

This initial release implements the complete application flow, but **a new full real-model run through the packaged application is deferred**. The owner asked to continue other work and postpone real Qwen validation when available memory was below the established 4.1 GiB startup threshold. No weaker memory limit was substituted to force a run.

Earlier private workflow experiments on this Mac established the staged Qwen3-TTS approach. Those results informed the implementation; they do **not** count as end-to-end verification of this new application, and the recordings/outputs are not distributed.

## Automated and integration checks

The public test suite uses synthetic video, test tones and explicitly substituted model output. Real components exercised include scene validation, SQLite transactions, immutable approval, stale-job rejection, source import, FFmpeg, Pillow subtitles, origin/token checks, HTTP provider serialization, cancellation and memory preflight.

A synthetic end-to-end test exports English, Chinese and bilingual videos and verifies that changing caption language reuses narration. Each output is checked with ffprobe and fully decoded with FFmpeg. No simulated model response is exposed as a production provider option.

The ASR dependency boundary has a regression test: faster-whisper 1.2.1 expects the `metadata_errors` argument removed in PyAV 19. The compatible PyAV 17.1.0 and CTranslate2 4.8.0 versions are pinned in the ASR extra.

Exact final counts and packaging checks are updated at release time below.

## Browser checks

- Real private source clips imported through the local-path UI; five thumbnails, durations, dimensions and audio-presence flags checked.
- Separate synthetic browser harness exercises file upload, manual script editing, immutable approval, the subtitle single-choice dialog, editable Chinese translation, reference audio input and MP4 export.
- The synthetic harness replaces TTS with a test tone and is not included in the public application. It verifies interaction and rendering, not speaker similarity or model quality.
- Responsive settings and project navigation checked in the narrow in-app browser as well as the wider test view.

## Model verification still pending

- Local Ollama Qwen3-VL analysis and script generation through the new full UI workflow.
- New packaged MLX Qwen synthesis, generated-speech ASR alignment and final rendering with real user narration.
- Subjective speaker similarity, pronunciation and naturalness for the new application output.
- Windows, Linux, CUDA and the experimental PyTorch adapter have not been hardware-tested.

A connection/schema test against a local HTTP fixture proves request/response handling, not a real model's visual understanding. ASR content matching is not a speaker-similarity score.

## Release checks

Pending final verification and release audit; this section will be updated before publication.
