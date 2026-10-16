# v0.1.1 verification record

Date: 2026-10-16. Primary development machine: Apple M3, 16 GB unified memory, macOS 26.5.2. Python 3.12.14 for the app; existing staged Qwen environment uses Python 3.11. FFmpeg 9.0.2 is the locally tested renderer.

## Real local vision and script verification

Ollama 0.34.3 with `qwen3-vl:2b` was tested against five private short clips through the application. All five visual analyses completed, source-audio checking ran, and the final job produced five editable scenes with valid source IDs and source-relative ranges. The browser showed the completed task and enabled script review. No script was approved and no TTS or final rendering job was started in this run. The existing 4.1 GiB startup and runtime memory guards remained enabled; no resource-limit event occurred.

The initial empty-content failure was reproduced: JSON appeared in Ollama's `thinking` field despite `think: false`, and unbounded repetition reached the token limit. A controlled single-clip comparison with a bounded schema returned complete JSON. The application now accepts a standalone JSON object from that field only on a completed normal stop, then applies the normal task validator. Script generation also reproduced cumulative movie timestamps exceeding individual source durations; source-relative prompting and a per-source schema resolved that rejection in the final application retry.

This confirms the request/response and draft-generation path, **not editorial quality**. The small local model's draft still needed factual corrections and shorter narration. Human review remains necessary. The footage, raw responses and browser proof for this run stay in private local app data and are not distributed.

## What the checks mean

The application implements the complete flow, but **a full real-model run including Qwen voice synthesis through the packaged application is deferred**. The owner asked to continue other work and postpone real Qwen validation when available memory was below the established 4.1 GiB startup threshold. No weaker memory limit was substituted to force a run. Local vision and script generation were subsequently verified as described above.

Earlier private workflow experiments on this Mac established the staged Qwen3-TTS approach. Those results informed the implementation; they do **not** count as end-to-end verification of this new application, and the recordings/outputs are not distributed.

## Automated and integration checks

The public test suite uses synthetic video, test tones and explicitly substituted model output. Real components exercised include scene validation, SQLite transactions, immutable approval, stale-job rejection, source import, FFmpeg, Pillow subtitles, origin/token checks, HTTP provider serialization, cancellation and memory preflight.

A synthetic end-to-end test exports English, Chinese and bilingual videos and verifies that changing caption language reuses narration. Each output is checked with ffprobe and fully decoded with FFmpeg. No simulated model response is exposed as a production provider option.

The ASR dependency boundary has a regression test: faster-whisper 1.2.1 expects the `metadata_errors` argument removed in PyAV 19. The compatible PyAV 17.1.0 and CTranslate2 4.8.0 versions are pinned in the ASR extra.

The final local suite passed **61 Python tests and 5 Node frontend tests**. Provider regressions cover completed JSON in the alternate Ollama field, rejecting prose and truncated output, bounded visual schemas, per-source script ranges and recovery after malformed response envelopes. The frontend regressions use deliberately delayed responses to check upload ownership, reference transcript capture, navigation order, script saves and completed-job refreshes. Cache tests interrupt extraction and corrupt completion markers. Streaming-upload tests exercise limits with and without Content-Length, early termination, cleanup, duplicate names and recognizable media names.

One independent whole-branch review found four Important issues and no Critical issues. All four were reproduced and fixed before release: asynchronous project isolation, interrupted-cache recovery, receive-time upload limits and filename retention. No review finding was deferred.

## Browser checks

- Real private source clips imported through the local-path UI; five thumbnails, durations, dimensions and audio-presence flags checked.
- Separate synthetic browser harness exercises file upload, manual script editing, immutable approval, the subtitle single-choice dialog, editable Chinese translation, reference audio input and MP4 export.
- The synthetic harness replaces TTS with a test tone and is not included in the public application. It verifies interaction and rendering, not speaker similarity or model quality.
- Responsive settings and project navigation checked in the narrow in-app browser as well as the wider test view.
- Repeated the complete synthetic browser flow after review fixes. The downloaded MP4 contains H.264 video at 1280 × 720 and AAC audio, duration 2.000 seconds. Uploaded `sample.mp4` retains its visible name.

![Synthetic workflow verification; test tone is not a cloned voice](ui-verification.png)

## Model verification still pending

- New packaged MLX Qwen synthesis, generated-speech ASR alignment and final rendering with real user narration.
- Subjective speaker similarity, pronunciation and naturalness for the new application output.
- Windows, Linux, CUDA and the experimental PyTorch adapter have not been hardware-tested.

A connection/schema test against a local HTTP fixture proves request/response handling, not a real model's visual understanding. ASR content matching is not a speaker-similarity score.

## Release checks

- Python suite: **61 passed**, including the installed ASR decoder boundary. The current Starlette/httpx test-client combination emits one deprecation warning; it does not affect the runtime or these results.
- Node 22-compatible frontend suite: **5 passed**. Ruff undefined/import checks and JavaScript syntax validation pass.
- Wheel built from the source distribution in an isolated build environment. The resulting wheel was installed into a separate clean Python environment and exercised outside the checkout: CLI, HTML-referenced static assets, project API/persistence, worker files and bundled font/license passed.
- Apple Silicon MLX + ASR extras resolve together with the pinned dependency versions. This dependency-resolution check does not load the TTS model.
- Public source candidates, historical text blobs, wheel and source archive were audited for private media/model artifacts, credentials and personal absolute paths; no such content was found. The screenshot uses synthetic material only.
- GitHub Actions repeats the Python suite, frontend regressions, Ruff and clean-wheel checks on Linux with Python 3.11 and 3.12. The CI status is displayed in GitHub; this is separate from the local results above.

This is distributed as a **preview prerelease**, with the real-model and hardware limitations above kept visible in both READMEs and release notes.
