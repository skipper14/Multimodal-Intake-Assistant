# AfyaPlus Multimodal Intake Assistant

## Overview

AfyaPlus is a safety-focused health intake and routing prototype. It accepts
health-related text and is being extended toward image and audio intake. The
chosen domain is Kenyan health education, triage, and insurance verification.

The current multimodal milestone is the image retrieval path:

1. Fifteen AfyaPlus health-poster images are stored in `week_5 /images/`.
2. OpenCLIP `ViT-B-32` encodes each image once into an L2-normalised vector.
3. A text query is encoded with the same CLIP model and ranked by cosine
   similarity.
4. Matches below the minimum similarity threshold of **0.20** are discarded
   and should be routed for review or returned as `UNREADABLE`.

The course's primary cloud path is GPT-4o vision plus Whisper. This repository
uses the equivalent open-source CLIP path for the completed image-index
milestone because no active OpenAI key is required. The threshold and safety
rules are unchanged by the model provider.

The captioning path is implemented in `caption_image.py`. It prefers an
OpenRouter-compatible GPT-4o vision model when `OPENROUTER_API_KEY` is set,
then falls back to `OPENAI_API_KEY` and local OpenCLIP when cloud access is not
available. Its constrained JSON contract classifies
each image as `CLEAR`, `AMBIGUOUS`, `OUT_OF_SCOPE`, or `UNREADABLE`. Missing
keys, quota errors, malformed model output, and unreadable files fail closed.

The transcription path is implemented in `transcribe_audio.py`. It uses
OpenAI Whisper (`whisper-1`) when `OPENAI_API_KEY` is available, otherwise it
uses local `faster-whisper`. A second structured extraction step returns
language, patient context, symptoms, medication/treatment, urgency, routing,
confidence, and human-review fields. The offline extractor is intentionally
conservative around multilingual respiratory emergency phrases.

## Project Structure

### Multimodal intake path

- `week_5 /images/`: Fifteen reproducible health-poster images used for
  retrieval tests.
- `week_5 /generate_demo_images.py`: Creates the local 15-image corpus.
- `week_5 /build_index.py`: Builds the CLIP index.
- `search.py`: Public `search(query, top_k)` and `build_index(...)` wrapper.
- `week_5 /image_index.pkl`: Generated vector index; rebuild it when images
  change.
- `caption_image.py`: OpenRouter/OpenAI GPT-4o vision captioning with a
  domain-constrained prompt, JSON normalization, disclaimer, and review gate.
- `week_5 /generate_caption_test_images.py`: Creates ambiguous and out-of-scope
  captioning fixtures.
- `transcribe_audio.py`: Whisper or faster-whisper transcription followed by
  structured AfyaPlus JSON extraction.
- `week_5 /audio_tests/french_emergency.wav`: Reproducible accented French
  sample used to test multilingual transcription and emergency routing.
- `app.py`: Existing AfyaPlus cloud/local text inference entry point.
- `app.py --ui`: Launches the unified Gradio image/audio application.
- `router.py`: Routes one uploaded file to the image captioner or audio
  transcription pipeline based on its extension.
- `afya_agent.py`: Grounded retrieval, PII masking, tool use, and routing
  agent for policy and insurance workflows.
- `defensive_gateway.py`: Prompt-injection-aware administrative/medical gate.

The existing text routing and safety behavior lives in `app.py`,
`afya_agent.py`, and `defensive_gateway.py`.

## How to Reproduce

Use a Python version with a compatible PyTorch wheel. The validated macOS
runtime is `/usr/local/bin/python3.14` with Torch 2.14, OpenCLIP 3.3, and
Pillow installed. The Homebrew Python 3.15 preview currently has no compatible
PyTorch wheel.

```bash
# From the repository root
/usr/local/bin/python3.14 -m pip install -r requirements.txt

# Generate or refresh the 15-image corpus
/usr/local/bin/python3.14 'week_5 /generate_demo_images.py'

# Build the index and run a retrieval query
/usr/local/bin/python3.14 'week_5 /build_index.py' \
  --query 'poster about malaria prevention' --top-k 3
```

Expected retrieval behavior is that `poster_01.png` and `poster_02.png`
(malaria prevention and malaria symptoms) appear first with the generated
corpus. The index records the threshold in `image_index.pkl`.

The existing text intake entry point can be run with:

```bash
/usr/local/bin/python3.14 app.py \
  'I have severe chest pain and shortness of breath.'
```

With `OPENAI_API_KEY` configured, `app.py` attempts the OpenAI path and then
falls back to local Ollama when cloud inference is unavailable. The local
fallback requires an installed Ollama model named `llama3.2`.

To run the captioner on the clear poster and the two safety fixtures:

```bash
/usr/local/bin/python3.14 'week_5 /generate_caption_test_images.py'
/usr/local/bin/python3.14 caption_image.py \
  'week_5 /images/poster_01.png' \
  'week_5 /caption_tests/ambiguous_blurred.png' \
  'week_5 /caption_tests/out_of_scope.png'
```

For OpenRouter, set `OPENROUTER_API_KEY` and optionally
`OPENROUTER_VISION_MODEL` (default: `openai/gpt-4o`) in `.env`. Every returned
object includes the disclaimer. Every safety-sensitive, ambiguous, unreadable,
or out-of-scope result also sets `human_review_required: true`.

To reproduce the transcription test:

```bash
# Uses OpenAI Whisper when OPENAI_API_KEY is configured.
/usr/local/bin/python3.14 transcribe_audio.py \
  'week_5 /audio_tests/french_emergency.wav'

# Force the local fallback for an offline test. The first run downloads the
# selected model; `small` gives the most accurate local result on this sample.
OPENAI_API_KEY= FASTER_WHISPER_MODEL=small \
  /usr/local/bin/python3.14 transcribe_audio.py \
  'week_5 /audio_tests/french_emergency.wav'
```

The French sample is detected as `fr`; its noisy local transcription still
routes likely breathing difficulty to `emergency_department` and sets
`human_review_required: true`. A failed or empty transcription returns an
empty transcript, `human_review`, and the disclaimer rather than guessed text.

For the scored reproduction, `faster-whisper` with `FASTER_WHISPER_MODEL=small`
returns an accurate French transcript with the emergency symptoms structured
as `douleur à la poitrine` and `difficultés à respirer`.

## Unified Gradio Application

Launch the browser interface from the repository root:

```bash
/usr/local/bin/python3.14 app.py --ui
```

Open the printed local URL, upload exactly one image or audio recording, and
select **Analyze uploaded file**. The interface labels the route as either
**Image sent to visual captioning** or **Audio sent to transcription and
structured extraction**. It shows the uploaded media, structured JSON output,
and the safety note in the same view.

Evidence screenshots from the launched application:

[![Gradio image route result](https://github.com/skipper14/Multimodal-Intake-Assistant/raw/main/week_5%20/ui_image_result.png)](https://github.com/skipper14/Multimodal-Intake-Assistant/blob/main/week_5%20/ui_image_result.png)
[![Gradio audio route result](https://github.com/skipper14/Multimodal-Intake-Assistant/raw/main/week_5%20/ui_audio_result.png)](https://github.com/skipper14/Multimodal-Intake-Assistant/blob/main/week_5%20/ui_audio_result.png)

## Evidence Screenshots

The repository includes visual corpus evidence that can be opened directly:

[![Malaria prevention retrieval corpus image](https://github.com/skipper14/Multimodal-Intake-Assistant/raw/main/week_5%20/images/poster_01.png)](https://github.com/skipper14/Multimodal-Intake-Assistant/blob/main/week_5%20/images/poster_01.png)
[![Malaria symptoms retrieval corpus image](https://github.com/skipper14/Multimodal-Intake-Assistant/raw/main/week_5%20/images/poster_02.png)](https://github.com/skipper14/Multimodal-Intake-Assistant/blob/main/week_5%20/images/poster_02.png)

The successful reproduction output is also observable from the build command:

```text
Indexed 15 images into .../week_5 /image_index.pkl
0.3349  .../images/poster_01.png
0.2835  .../images/poster_02.png
```

## Safety Note

- Image captions and retrieval matches are informational and operational only;
  they are **not a medical diagnosis**.
- A similarity match is not proof that an image is clinically relevant.
- Low-confidence or unreadable inputs must not be guessed at. Return
  `UNREADABLE` or route to qualified human review.
- Urgent symptoms such as severe chest pain, breathing difficulty, heavy
  bleeding, or loss of consciousness require immediate emergency escalation.
- Do not expose private patient information unnecessarily; the text agent masks
  Kenyan phone numbers and email addresses before model processing.

## Evaluation and Recommendation

The capability-level before/after evidence is recorded in
`evaluation/multimodal_before_after.csv`. The stakeholder decision memo is
`stakeholder_memo.md`. The recommendation is **go for a small, human-reviewed
pilot**, but **no-go for autonomous diagnosis or clinical decisions** until
live vision-provider validation and additional reviewer testing are complete.
