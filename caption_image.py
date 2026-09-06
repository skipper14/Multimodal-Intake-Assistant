"""Safety-gated AfyaPlus image captioning with OpenRouter/OpenAI vision."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageStat

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')

DISCLAIMER = (
    'Informational only; this image description is not a medical diagnosis. '
    'Seek qualified human care for symptoms or urgent concerns.'
)
SYSTEM_PROMPT = """You are the AfyaPlus health-image captioning component.
Return ONLY one valid JSON object with exactly these keys:
status, caption, visible_text, domain_relevance, safety_sensitive,
human_review_required, reason.

Allowed status values: CLEAR, AMBIGUOUS, OUT_OF_SCOPE, UNREADABLE.
Allowed domain_relevance values: health_education, clinical_intake,
out_of_scope, unknown.

Describe only visible, non-diagnostic facts. Do not identify a disease,
interpret a scan, estimate severity, recommend treatment, infer age/gender,
or identify a person. If text or content is unclear, use AMBIGUOUS or
UNREADABLE instead of guessing. Use OUT_OF_SCOPE for non-health content.
Set safety_sensitive true for symptoms, injuries, medication, pregnancy,
diagnostic-looking material, or any content that could influence care.
Set human_review_required true for every safety-sensitive, ambiguous,
unreadable, or out-of-scope response."""


@dataclass
class CaptionResult:
    status: str
    caption: str
    visible_text: list[str]
    domain_relevance: str
    safety_sensitive: bool
    human_review_required: bool
    reason: str
    disclaimer: str = DISCLAIMER


def _safe_result(status: str, reason: str) -> CaptionResult:
    return CaptionResult(
        status=status,
        caption='No reliable caption was produced.',
        visible_text=[],
        domain_relevance='unknown' if status != 'OUT_OF_SCOPE' else 'out_of_scope',
        safety_sensitive=status in {'AMBIGUOUS', 'UNREADABLE'},
        human_review_required=True,
        reason=reason,
    )


def _image_data_url(image_path: Path) -> str:
    if not image_path.is_file():
        raise FileNotFoundError(f'Image not found: {image_path}')
    mime_type = mimetypes.guess_type(image_path.name)[0] or 'image/png'
    encoded = base64.b64encode(image_path.read_bytes()).decode('ascii')
    return f'data:{mime_type};base64,{encoded}'


def _client_and_model() -> tuple[OpenAI | None, str, str | None]:
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    if openrouter_key:
        return (
            OpenAI(
                api_key=openrouter_key,
                base_url='https://openrouter.ai/api/v1',
                default_headers={
                    'HTTP-Referer': 'https://afyaplus.example',
                    'X-Title': 'AfyaPlus Multimodal Intake',
                },
            ),
            os.getenv('OPENROUTER_VISION_MODEL', 'openai/gpt-4o'),
            'openrouter',
        )
    if openai_key:
        return OpenAI(api_key=openai_key), os.getenv('OPENAI_VISION_MODEL', 'gpt-4o'), 'openai'
    return None, '', None


@lru_cache(maxsize=1)
def _local_clip_components():
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    return model.eval(), preprocess, open_clip.get_tokenizer('ViT-B-32'), torch


def _local_caption(image_path: Path) -> CaptionResult:
    image = Image.open(image_path).convert('RGB')
    if max(ImageStat.Stat(image).stddev) < 18:
        return _safe_result('AMBIGUOUS', 'The image contains too little visual detail for a reliable caption.')

    topics = [
        'malaria prevention', 'malaria symptoms', 'nutrition and balanced diet',
        'child nutrition', 'maternal antenatal care', 'newborn care',
        'first aid and emergency care', 'wound care', 'hygiene and hand washing',
        'safe drinking water', 'diabetes management', 'blood pressure screening',
        'vaccination schedule', 'mental health support',
        'sexual and reproductive health',
    ]
    labels = [f'a health education poster about {topic}' for topic in topics]
    labels.append('a non-health landscape or general illustration')
    model, preprocess, tokenizer, torch = _local_clip_components()
    with torch.no_grad():
        image_vector = model.encode_image(preprocess(image).unsqueeze(0))
        image_vector = image_vector / image_vector.norm(dim=-1, keepdim=True)
        text_vector = model.encode_text(tokenizer(labels))
        text_vector = text_vector / text_vector.norm(dim=-1, keepdim=True)
        scores = (image_vector @ text_vector.T).squeeze(0)
    health_score = float(scores[:-1].max())
    out_of_scope_score = float(scores[-1])
    if out_of_scope_score >= health_score:
        return CaptionResult(
            status='OUT_OF_SCOPE',
            caption='No health-related content was identified in this image.',
            visible_text=[], domain_relevance='out_of_scope', safety_sensitive=False,
            human_review_required=True, reason='The local vision model matched a non-health category.',
        )
    topic_index = int(scores[:-1].argmax())
    return CaptionResult(
        status='CLEAR',
        caption=f'Visible health-education poster about {topics[topic_index]}.',
        visible_text=[], domain_relevance='health_education', safety_sensitive=True,
        human_review_required=True, reason='Local CLIP topic match; verify against the original image.',
    )


def _normalise_response(payload: Any) -> CaptionResult:
    if not isinstance(payload, dict):
        return _safe_result('UNREADABLE', 'Vision model returned a non-object response.')

    status = str(payload.get('status', 'UNREADABLE')).upper()
    if status not in {'CLEAR', 'AMBIGUOUS', 'OUT_OF_SCOPE', 'UNREADABLE'}:
        status = 'UNREADABLE'
    relevance = str(payload.get('domain_relevance', 'unknown'))
    if relevance not in {'health_education', 'clinical_intake', 'out_of_scope', 'unknown'}:
        relevance = 'unknown'
    visible_text = payload.get('visible_text', [])
    if isinstance(visible_text, str):
        visible_text = [visible_text]
    if not isinstance(visible_text, list):
        visible_text = []
    safety_sensitive = bool(payload.get('safety_sensitive', False))
    review_required = bool(payload.get('human_review_required', False))
    review_required = review_required or safety_sensitive or status != 'CLEAR'
    return CaptionResult(
        status=status,
        caption=str(payload.get('caption') or 'No reliable caption was produced.'),
        visible_text=[str(item) for item in visible_text[:20]],
        domain_relevance=relevance,
        safety_sensitive=safety_sensitive,
        human_review_required=review_required,
        reason=str(payload.get('reason') or 'No reason supplied.'),
    )


def caption_image(image_path: str | Path, client: OpenAI | None = None, model: str | None = None) -> dict[str, Any]:
    """Caption one image and always return the disclaimer/review fields."""
    image_path = Path(image_path)
    try:
        image_url = _image_data_url(image_path)
    except (FileNotFoundError, OSError) as error:
        return asdict(_safe_result('UNREADABLE', str(error)))

    selected_client = client
    selected_model = model
    if selected_client is None:
        selected_client, selected_model, provider = _client_and_model()
        if selected_client is None:
            try:
                result = _local_caption(image_path)
                output = asdict(result)
                output['provider'] = 'local-openclip'
                output['model'] = 'ViT-B-32'
                return output
            except Exception as error:
                return asdict(_safe_result(f'UNREADABLE', f'Local vision captioning unavailable: {error}'))
    else:
        provider = 'injected-client'

    try:
        response = selected_client.chat.completions.create(
            model=selected_model,
            temperature=0,
            max_tokens=350,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Caption this image under the AfyaPlus rules.'},
                    {'type': 'image_url', 'image_url': {'url': image_url}},
                ]},
            ],
        )
        content = response.choices[0].message.content or '{}'
        result = _normalise_response(json.loads(content))
        output = asdict(result)
        output['provider'] = provider
        output['model'] = selected_model
        return output
    except Exception as error:
        return asdict(_safe_result('UNREADABLE', f'Vision captioning failed safely: {error}'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='+', type=Path)
    args = parser.parse_args()
    for image_path in args.images:
        print(json.dumps({'image': str(image_path), **caption_image(image_path)}, indent=2))


if __name__ == '__main__':
    main()