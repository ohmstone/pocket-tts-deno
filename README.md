# Pocket TTS — Deno Server Design

[Main repo](https://codeberg.org/ohmstone/pocket-tts-deno)
[Mirror](https://github.com/ohmstone/pocket-tts-deno)

## Overview

This project is a Deno server port of
[pocket-tts-server](https://github.com/ai-joe-git/pocket-tts-server). The prior
project is itself a webapp (wasm + onnx) build of
[pocket-tts](https://github.com/kyutai-labs/pocket-tts). Pocket TTS is a very
good quality, small, TTS model that includes voice cloning.

The intent of this port-of-a-port is to make it very fast to add a decent
CPU-based, minimal RAM, TTS tool to cloud and local apps. It borrows ideas from
other similar ports where it uses a common API design. Feel free to fork the
project and adapt it to your needs, this reuses the open licenses of the prior
projects.

**NOTE** This port was done mostly automatically with Claude. I've done what I
can to review the implementation, but it was minimal at best. Therefore, read
the code and reconsider before putting this into anything production-facing. Or,
in other words: **USE AT YOUR OWN RISK**!

`server.ts` is a Deno server that exposes a REST API for neural text-to-speech
(TTS) with voice cloning. It runs the full inference pipeline locally on CPU
using ONNX Runtime — no GPU, no cloud, no external services.

The server is a direct port of the browser web worker (`inference-worker.js`)
into a server context, reusing the same ONNX models, tokenizer, and voices. The
SentencePiece tokenizer is provided by `sentencepiece.ts`, a standalone
TypeScript module that loads the SentencePiece WebAssembly binary
(`sentencepiece.wasm`) at runtime. `server.ts` imports from it directly.

---

## Files

### Source files

| File                 | Purpose                                                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `server.ts`          | Deno HTTP server — entry point for the API and full inference pipeline                                                                     |
| `sentencepiece.ts`   | SentencePiece tokenizer module; loads `sentencepiece.wasm` and exposes `createSentencePieceModule`                                         |
| `sentencepiece.wasm` | WebAssembly binary for the SentencePiece tokenizer (~486 KB)                                                                               |
| `deno.json`          | Deno config: sets `"nodeModulesDir": "auto"`, maps `"ort"` → `onnxruntime-node` and `"module"` → `node:module`, and defines a `serve` task |
| `README.md`          | This document                                                                                                                              |

### Data files required at runtime

| File              | Purpose                                                 |
| ----------------- | ------------------------------------------------------- |
| `tokenizer.model` | SentencePiece model data (~60 KB)                       |
| `voices.bin`      | Packed float32 embeddings for built-in voices (~1.5 MB) |
| `onnx/*.onnx`     | Five ONNX model files (~190 MB total)                   |

---

## How the System Works

### Model Architecture

Pocket TTS uses five ONNX models working in series to convert text to audio:

```
Text
  │
  ▼
[text_conditioner]   → text embeddings (token IDs → 1024-dim vectors)
  │
  ▼
[flow_lm_main]       → autoregressive transformer
  │                    Runs twice per step:
  │                    1. Text conditioning pass (updates KV-cache state)
  │                    2. AR generation pass (predicts one 32-dim latent + EOS logit)
  │
  ▼  (flow matching Euler integration)
[flow_lm_flow]       → refines each latent via stochastic flow matching
  │                    Runs LSD times per latent (LSD = Latent Solver/Diffusion steps)
  │
  ▼
[mimi_decoder]       → decodes latent frames → raw audio samples
  │                    Streaming: decodes in batches as latents accumulate
  │
  ▼
PCM audio @ 24 kHz
```

Voice conditioning uses a separate model:

```
Reference audio (Float32, 24 kHz mono)
  │
  ▼
[mimi_encoder]       → voice embedding [1, frames, 1024]
  │
  ▼
[flow_lm_main]       → voice-conditioned KV-cache state (cached per voice)
```

### Inference Pipeline

1. **Text preprocessing** — Numbers, abbreviations, special characters, and
   unicode are normalised to plain ASCII speech text.

2. **Chunking** — Text is split into sentence-level chunks targeting ≤50
   SentencePiece tokens each, so the AR model never exceeds its context limit.

3. **For each chunk:**
   - Text is tokenised and passed through `text_conditioner` to get embeddings.
   - A text conditioning pass through `flow_lm_main` primes the KV-cache.
   - The AR loop generates one 32-dimensional latent per step. At each step:
     - `flow_lm_main` produces a conditioning vector and an EOS logit.
     - `flow_lm_flow` refines a Gaussian noise sample into the final latent via
       Euler integration (LSD steps, default 10).
     - Generation ends after detecting EOS and running `FRAMES_AFTER_EOS` = 3
       extra steps (matching the PyTorch reference behaviour).
   - Latents are decoded in streaming batches (3 frames first, then 12 frames at
     a time) by `mimi_decoder` to minimise time-to-first-audio.
   - Between chunks a short silence gap (250 ms) is inserted.

4. **State management** — Flow LM and MIMI decoder maintain internal KV-cache
   and convolutional states across steps. States are reset per chunk to avoid
   artefact accumulation.

5. **Voice conditioning cache** — The KV-cache state resulting from a voice
   embedding pass is cached in memory per voice name. Subsequent requests using
   the same voice skip this expensive step.

### Audio Output

Each MIMI decoder output is a Float32Array of raw PCM at 24 000 Hz, mono. The
server converts samples to 16-bit PCM and streams them as a WAV file.

The WAV header is written immediately with a streaming placeholder size
(`0xFFFFFFFF` in the RIFF and data chunk size fields), which signals to decoders
that the length is unknown. The response uses `Transfer-Encoding:
chunked`. Most
players and libraries (ffmpeg, VLC, Python `requests`, curl) handle this
correctly.

---

## Usage

### Prerequisites

- [Deno](https://deno.land/) v2.x

No other installation steps. Deno fetches `onnxruntime-node` automatically on
first run and caches it.

### Start the server

```bash
deno run \
  --allow-read \
  --allow-net \
  --allow-sys \
  --allow-ffi \
  --allow-env \
  server.ts
```

The server binds to a random available port and prints it at startup:

```
Pocket TTS — Deno Server
Loading models...
Models ready.

Server listening on http://localhost:54321
  Voices: cosette, jean, fantine
  Try: curl -s http://localhost:54321/v1/audio/speech \
         -H 'Content-Type: application/json' \
         -d '{"input":"Hello world!","voice":"cosette"}' > speech.wav
```

You can also pass `--port <port number>` after `server.ts` to use a
specific port.

### Generate speech

```bash
# Save to file
curl -s http://localhost:PORT/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hello, world!", "voice": "cosette"}' \
  -o speech.wav

# Stream directly to a player (requires ffplay from FFmpeg)
curl -s http://localhost:PORT/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hello, world!", "voice": "cosette"}' \
  | ffplay -nodisp -autoexit -i -
```

### List voices

```bash
curl http://localhost:PORT/v1/voices
```

```json
{
  "voices": ["cosette", "jean", "fantine"],
  "builtin": ["cosette", "jean", "fantine"],
  "custom": []
}
```

### Register a custom voice (voice cloning)

Upload a WAV file with a few seconds of clear speech. Any sample rate and
channel count is accepted; the server resamples to 24 kHz mono internally. Only
WAV files are supported (no MP3/AAC without additional tooling).

```bash
# Multipart upload
curl -s http://localhost:PORT/v1/voices?name=alice \
  -F "file=@sample.wav" \
  -o /dev/null -w "%{http_code}\n"

# Or raw WAV body
curl -s http://localhost:PORT/v1/voices?name=alice \
  -H 'Content-Type: audio/wav' \
  --data-binary @sample.wav \
  -o /dev/null -w "%{http_code}\n"
```

Response:

```json
{
  "id": "alice",
  "frames": 42,
  "status": "ready"
}
```

Then generate with the cloned voice:

```bash
curl -s http://localhost:PORT/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hello from my cloned voice!", "voice": "alice"}' \
  -o cloned.wav
```

### Remove a custom voice

```bash
curl -X DELETE http://localhost:PORT/v1/voices/alice
```

### Python example

```python
import requests

BASE = "http://localhost:PORT"

# Generate speech
r = requests.post(f"{BASE}/v1/audio/speech", json={
    "input": "This is Pocket TTS running in Deno.",
    "voice": "cosette",
}, stream=True)

with open("output.wav", "wb") as f:
    for chunk in r.iter_content(chunk_size=4096):
        f.write(chunk)

# Clone a voice
with open("reference.wav", "rb") as f:
    r = requests.post(f"{BASE}/v1/voices?name=myvoice", files={"file": f})
print(r.json())  # {"id": "myvoice", "frames": ..., "status": "ready"}
```

---

## API Reference

### `POST /v1/audio/speech`

Generate speech from text. Returns a streaming WAV audio response.

**Request body (JSON):**

| Field             | Type   | Default     | Description                                        |
| ----------------- | ------ | ----------- | -------------------------------------------------- |
| `input`           | string | required    | Text to synthesise                                 |
| `voice`           | string | `"cosette"` | Voice name (built-in or registered custom)         |
| `speed`           | number | 1.0         | Accepted for API compatibility, currently ignored  |
| `response_format` | string | `"wav"`     | Accepted for API compatibility, always returns WAV |

**Response:** `audio/wav`, chunked transfer encoding, 24 kHz mono PCM16.

---

### `GET /v1/voices`

List available voices.

**Response:**

```json
{
  "voices": ["cosette", "jean", "fantine", "alice"],
  "builtin": ["cosette", "jean", "fantine"],
  "custom": ["alice"]
}
```

---

### `POST /v1/voices`

Register a custom voice from a WAV audio sample.

**Query parameters:**

- `name` (optional) — Name to assign the voice. Defaults to a timestamp-based
  ID.

**Request body:** WAV audio, either as:

- `multipart/form-data` with a `file` field, or
- raw `audio/wav` bytes

**Constraints:**

- Audio is truncated to 10 seconds
- Stereo is mixed to mono
- Any sample rate is resampled to 24 kHz
- Only WAV format is accepted (PCM16, PCM32, or IEEE float)

**Response (201):**

```json
{ "id": "alice", "frames": 42, "status": "ready" }
```

---

### `DELETE /v1/voices/:name`

Remove a registered custom voice. Built-in voices cannot be deleted.

---

### `GET /health`

Returns `200 OK` when models are loaded and ready, or `503` while loading.

---

### `GET /`

Returns server info and a summary of available endpoints.

---

## Design Decisions

**Single file.** All inference logic, text preprocessing, WAV encoding, and HTTP
routing live in `server.ts`. The only external dependency is `onnxruntime-node`,
pulled via Deno's npm import. No package.json, no bundler, no build step.

**Streaming WAV.** Audio is written to the HTTP response as it is generated,
frame by frame, rather than waiting for full generation to finish. The WAV
header uses `0xFFFFFFFF` as a placeholder for the data size (the streaming WAV
convention). Clients can save the response body directly to a `.wav` file or
pipe it to a player.

**Voice conditioning cache.** The most expensive per-voice operation (running
the voice embedding through the flow LM to initialise the KV-cache) is cached in
memory. The first request for a voice pays this cost; subsequent requests start
generation immediately. The built-in default voice (`cosette`) is
pre-conditioned at startup.

**In-memory custom voices.** Custom voice embeddings and their conditioned
states are stored in memory and lost when the server restarts. This matches the
browser behaviour and keeps the server stateless. For persistence, embeddings
could be saved to disk with minor additions.

**CPU execution.** `onnxruntime-node` is configured with the `"cpu"` execution
provider and `intraOpNumThreads` set to the number of hardware cores. This uses
ONNX Runtime's native optimised kernels. The INT8-quantised flow LM models run
significantly faster than their float32 equivalents.

**WAV-only voice cloning input.** Decoding arbitrary audio formats (MP3, AAC,
etc.) without external libraries requires codec implementations. Restricting to
WAV keeps the server dependency-free. Users can convert audio with
`ffmpeg -i input.mp3 output.wav` before uploading.

## License

- **Models & Voice Embeddings**: CC BY 4.0 (inherited from
  [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts))
- **Original Code**: Apache 2.0
