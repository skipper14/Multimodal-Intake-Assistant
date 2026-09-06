"""Unified image/audio input router for the AfyaPlus Gradio application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from caption_image import caption_image
from transcribe_audio import transcribe_and_extract

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.ogg', '.webm', '.flac', '.aiff'}


def classify_file(file_path: str | Path) -> str:
    """Classify an uploaded path without inspecting its content."""
    suffix = Path(file_path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return 'image'
    if suffix in AUDIO_EXTENSIONS:
        return 'audio'
    return 'unsupported'


def route_file(file_path: str | Path | None) -> dict[str, Any]:
    """Dispatch one upload to the image or audio pipeline."""
    if not file_path:
        return {
            'input_type': 'none',
            'status': 'Upload an image or audio recording to begin.',
            'result': {},
        }
    path = Path(file_path)
    input_type = classify_file(path)
    if input_type == 'image':
        result = caption_image(path)
        return {'input_type': 'image', 'status': 'Image sent to visual captioning.', 'result': result}
    if input_type == 'audio':
        result = transcribe_and_extract(path)
        return {'input_type': 'audio', 'status': 'Audio sent to transcription and structured extraction.', 'result': result}
    return {
        'input_type': 'unsupported',
        'status': 'Unsupported file type. Upload an image or audio recording.',
        'result': {},
    }


def route_for_gradio(file_path: str | None) -> tuple[str, str, str | None, str | None]:
    """Return UI-ready route label, JSON result, and media preview paths."""
    routed = route_file(file_path)
    input_type = routed['input_type']
    if input_type == 'image':
        return routed['status'], json.dumps(routed['result'], indent=2, ensure_ascii=False), str(file_path), None
    if input_type == 'audio':
        return routed['status'], json.dumps(routed['result'], indent=2, ensure_ascii=False), None, str(file_path)
    return routed['status'], json.dumps(routed['result'], indent=2), None, None