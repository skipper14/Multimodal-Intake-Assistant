"""AfyaPlus Whisper transcription and safe structured extraction."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')

DISCLAIMER = (
    'Informational intake support only; this is not a medical diagnosis. '
    'A qualified clinician must review symptoms and treatment decisions.'
)
EXTRACTION_PROMPT = """You are the AfyaPlus structured intake extractor.
Convert the transcript into ONLY one valid JSON object with exactly these keys:
language, patient_context, detected_symptoms, medication_or_treatment,
urgency, is_critical_emergency, routing_destination, extraction_confidence,
human_review_required, reason.

Use empty strings/lists when information is absent. Do not invent facts or
translate away uncertainty. urgency must be one of routine, soon, urgent,
emergency, or unknown. routing_destination must be one of primary_care,
pharmacy_review, appointment_booking, emergency_department, human_review, or
unknown. Set human_review_required true for emergency or urgent symptoms,
medication questions, missing/unclear content, or any uncertain extraction.
This is administrative structuring, not diagnosis or treatment advice."""


def _default_fields(reason: str = 'No transcript was available.') -> dict[str, Any]:
    return {
        'language': 'unknown',
        'patient_context': '',
        'detected_symptoms': [],
        'medication_or_treatment': '',
        'urgency': 'unknown',
        'is_critical_emergency': False,
        'routing_destination': 'human_review',
        'extraction_confidence': 0.0,
        'human_review_required': True,
        'reason': reason,
    }


def _normalise_fields(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return _default_fields('Extractor returned a non-object response.')
    result = _default_fields()
    for key in ('language', 'patient_context', 'medication_or_treatment', 'reason'):
        if isinstance(payload.get(key), str):
            result[key] = payload[key].strip()
    symptoms = payload.get('detected_symptoms', [])
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    if isinstance(symptoms, list):
        result['detected_symptoms'] = [str(item).strip() for item in symptoms if str(item).strip()]
    if payload.get('urgency') in {'routine', 'soon', 'urgent', 'emergency', 'unknown'}:
        result['urgency'] = payload['urgency']
    if payload.get('routing_destination') in {
        'primary_care', 'pharmacy_review', 'appointment_booking',
        'emergency_department', 'human_review', 'unknown'
    }:
        result['routing_destination'] = payload['routing_destination']
    try:
        result['extraction_confidence'] = max(0.0, min(1.0, float(payload.get('extraction_confidence', 0.0))))
    except (TypeError, ValueError):
        result['extraction_confidence'] = 0.0
    result['is_critical_emergency'] = bool(payload.get('is_critical_emergency', False))
    result['human_review_required'] = bool(payload.get('human_review_required', False))
    result['human_review_required'] = result['human_review_required'] or result['urgency'] in {'urgent', 'emergency'}
    if result['is_critical_emergency']:
        result['urgency'] = 'emergency'
        result['routing_destination'] = 'emergency_department'
        result['human_review_required'] = True
    return result


def _keyword_extract(transcript: str) -> dict[str, Any]:
    """Conservative offline extraction used when no structured LLM is available."""
    text = transcript.strip()
    lower = text.lower()
    emergency_terms = (
        'severe chest pain', 'shortness of breath', 'difficulty breathing', 'unconscious',
        'douleur à la poitrine', 'difficultés à respirer', 'difficulte a respirer',
        'maumivu ya kifua', 'kupumua kwa shida',
    )
    respiratory_emergency = 'respir' in lower and any(term in lower for term in ('difficult', 'short', 'souffle', 'shida'))
    symptoms = [
        phrase for phrase in (
            'fever', 'cough', 'chest pain', 'shortness of breath',
            'difficulty breathing', 'vomiting', 'dizziness', 'bleeding',
            'douleur à la poitrine', 'difficultés à respirer', 'maumivu ya kifua',
        ) if phrase in lower
    ]
    if respiratory_emergency and 'difficulty breathing' not in symptoms:
        symptoms.append('difficulty breathing')
    medication = ''
    medication_match = re.search(r'\b(?:taking|take|prescribed)\s+([^.,!?]+)', text, re.IGNORECASE)
    if medication_match:
        medication = medication_match.group(1).strip()
    is_emergency = any(term in lower for term in emergency_terms) or respiratory_emergency
    urgency = 'emergency' if is_emergency else ('soon' if symptoms else 'unknown')
    language = 'fr' if any(term in lower for term in ('bonjour', 'douleur', 'respirer')) else (
        'sw' if any(term in lower for term in ('maumivu', 'kupumua')) else 'unknown'
    )
    return _normalise_fields({
        'language': language,
        'patient_context': text[:500],
        'detected_symptoms': symptoms,
        'medication_or_treatment': medication,
        'urgency': urgency,
        'is_critical_emergency': is_emergency,
        'routing_destination': 'emergency_department' if is_emergency else ('human_review' if symptoms else 'unknown'),
        'extraction_confidence': 0.55 if symptoms else 0.25,
        'human_review_required': True,
        'reason': 'Offline keyword extraction; clinician review required.',
    })


def _cloud_client() -> OpenAI | None:
    api_key = os.getenv('OPENAI_API_KEY')
    return OpenAI(api_key=api_key) if api_key else None


def _transcribe_cloud(audio_path: Path, client: OpenAI) -> tuple[str, str, str | None]:
    with audio_path.open('rb') as audio_file:
        response = client.audio.transcriptions.create(
            model=os.getenv('OPENAI_TRANSCRIPTION_MODEL', 'whisper-1'),
            file=audio_file,
            response_format='verbose_json',
        )
    return getattr(response, 'text', '') or '', getattr(response, 'language', 'unknown') or 'unknown', 'openai-whisper'


def _transcribe_local(audio_path: Path) -> tuple[str, str, str | None]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError('faster-whisper is not installed') from error
    model = WhisperModel(os.getenv('FASTER_WHISPER_MODEL', 'small'), device='cpu', compute_type='int8')
    segments, info = model.transcribe(str(audio_path), beam_size=5)
    transcript = ' '.join(segment.text.strip() for segment in segments).strip()
    return transcript, getattr(info, 'language', 'unknown') or 'unknown', 'faster-whisper'


def extract_fields(transcript: str, client: OpenAI | None = None) -> dict[str, Any]:
    """Extract domain fields using JSON-mode LLM output or safe offline fallback."""
    if not transcript.strip():
        return _default_fields()
    client = client or _cloud_client()
    if client is None:
        return _keyword_extract(transcript)
    try:
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_EXTRACTION_MODEL', 'gpt-4o-mini'),
            temperature=0,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': EXTRACTION_PROMPT},
                {'role': 'user', 'content': transcript},
            ],
        )
        return _normalise_fields(json.loads(response.choices[0].message.content or '{}'))
    except Exception:
        return _keyword_extract(transcript)


def transcribe_and_extract(audio_path: str | Path, client: OpenAI | None = None) -> dict[str, Any]:
    """Transcribe audio and return a structured, reviewable intake record."""
    path = Path(audio_path)
    if not path.is_file():
        return {'audio_path': str(path), 'transcript': '', 'transcription_source': 'none',
                'language': 'unknown', 'fields': _default_fields(f'Audio not found: {path}'),
                'disclaimer': DISCLAIMER}
    client = client or _cloud_client()
    try:
        if client is not None:
            transcript, language, source = _transcribe_cloud(path, client)
        else:
            transcript, language, source = _transcribe_local(path)
    except Exception as error:
        transcript, language, source = '', 'unknown', 'failed'
        fields = _default_fields(f'Transcription failed safely: {error}')
    else:
        fields = extract_fields(transcript, client=client)
        if language != 'unknown':
            fields['language'] = language
    return {'audio_path': str(path), 'transcript': transcript,
            'transcription_source': source, 'language': language,
            'fields': fields, 'disclaimer': DISCLAIMER}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    args = parser.parse_args()
    print(json.dumps(transcribe_and_extract(args.audio), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()