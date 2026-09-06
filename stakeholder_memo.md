# Stakeholder Decision Memo: AfyaPlus Multimodal Intake

**To:** AfyaPlus Product, Clinical Safety, and Operations Leads  
**From:** Engineering  
**Date:** 6 September 2026  
**Decision requested:** Approve a controlled pilot of multimodal intake.

## Recommendation

**Go for a small, human-reviewed pilot. No-go for autonomous clinical decisions or diagnosis.**

The system now accepts an image or audio recording, chooses the correct processing path, and returns a structured result with safety language. The pilot should be limited to trained staff, approved test users, and monitored health-education and intake workflows. A clinician or trained reviewer must make the final decision whenever content is unclear, urgent, medication-related, or likely to affect care.

## What Changed

| Capability | Before | After | Evidence and decision |
|---|---|---|---|
| Image retrieval | No reusable image search | 15-image CLIP index, text search, 0.20 similarity gate | Malaria query ranked the two malaria posters first. **Pass for assisted retrieval.** |
| Image captioning | No vision caption path | Constrained GPT-4o-compatible captioner with local OpenCLIP fallback and safe status labels | Clear, ambiguous, and out-of-scope fixtures passed locally; every result includes a disclaimer and review flag. **Pass for assisted captioning.** |
| Audio transcription and extraction | No audio path | Whisper plus local faster-whisper fallback and structured fields | French sample detected as `fr`; breathing difficulty routed to emergency review. **Pass for assisted intake.** |

The detailed evidence is in `evaluation/multimodal_before_after.csv`.

## Main Safety Risk

**Risk: an image caption or speech transcript can be wrong or incomplete.** The local French speech model produced a noisy transcript. A wrong description could delay care or send a person to the wrong service.

**Mitigation:** the system fails closed for missing, unreadable, or low-confidence inputs; requires human review for urgent or uncertain content; routes detected breathing difficulty and similar red flags to emergency review; applies the image similarity threshold; and displays a plain-language disclaimer that the output is not a diagnosis. Reviewers must see the original image or play the original audio before acting.

## Pilot Conditions

1. Configure and test the approved vision provider before pilot traffic; the current OpenAI key has no remaining credits, so the validated submission path uses local OpenCLIP and keeps the cloud path optional.
2. Log route, model source, confidence, review outcome, and failure reason.
3. Review the first 50 image and 50 audio cases for missed red flags and false routing before expanding access.
4. Stop the pilot if any urgent case is not flagged for human review or if the original media is not available to the reviewer.

**Bottom line:** the engineering controls justify a constrained pilot that helps staff organise incoming media. They do not yet justify unsupervised medical advice, diagnosis, or emergency decisions.
