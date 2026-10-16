# Changelog

## 0.1.1 preview — 2026-10-16

Fix local Qwen3-VL analysis failing with an empty-content JSON error in Ollama 0.34.3. Scene analysis now requests a bounded JSON schema to prevent repetitive output exhausting the token budget. When Ollama places a completed JSON answer in its `thinking` field despite `think: false`, the adapter accepts only a complete standalone object after a normal stop and still applies the task validator. Prose, malformed content and truncated responses remain rejected; token-limit errors now explain the cause.

Analysis caches use a new contract version. Existing projects, approved scripts and media are preserved. Memory thresholds and the two-attempt retry limit are unchanged.

Script generation now explicitly uses offsets within each source video and supplies a schema tying each source ID to its own duration. Cumulative movie timestamps that exceed a source remain blocked by validation. Malformed provider response envelopes retain the bounded retry behavior.

Verified locally with five real source clips through Ollama analysis and editable script generation. The draft still needs human factual and pacing review; real Qwen narration-to-render validation remains pending. Automated checks: 61 Python tests and 5 frontend tests passed.

## 0.1.0 preview — 2026-10-16

Initial local browser application: multi-clip import, timestamped vision analysis, editable/reviewed script revisions, Chinese/English/bilingual captions, reference voice input, staged Qwen3-TTS MLX, local ASR alignment, FFmpeg MP4 rendering, durable tasks, cancellation and memory guards.

Uploads retain original display names, enforce limits while receiving, and remain bound to the originating project when navigating elsewhere. Interrupted or malformed analysis, transcription, audio and component caches are regenerated safely.

Apple Silicon is the primary verification target. The official PyTorch adapter is experimental. CI model operations are explicit test substitutes. A complete new real-model run through the packaged app is deferred; see the verification record.
