"""Build and query the AfyaPlus CLIP image retrieval index.

The index uses cosine similarity because both image and text embeddings are
L2-normalised. A score of 0.20 is the minimum acceptance threshold: below it,
the result is treated as unrelated and the caller must route to review. This
conservative threshold is deliberately lower than the usual top-result score
range for this small, synthetic corpus while still rejecting near-zero matches.
"""

import argparse
import pickle
from pathlib import Path

import open_clip
import torch
from PIL import Image

MODEL_NAME = 'ViT-B-32'
PRETRAINED = 'openai'
MIN_SIMILARITY = 0.20
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = HERE / 'images'
DEFAULT_INDEX_PATH = HERE / 'image_index.pkl'


def _load_model(device=None):
    selected_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model.to(selected_device).eval(), preprocess, tokenizer, selected_device


def build_index(image_dir=DEFAULT_IMAGE_DIR, index_path=DEFAULT_INDEX_PATH):
    """Encode all supported images under ``image_dir`` and persist the index."""
    image_dir = Path(image_dir)
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ) if image_dir.is_dir() else []
    if not image_paths:
        raise FileNotFoundError(f'No supported images found in {image_dir}')

    model, preprocess, _, device = _load_model()
    vectors, valid_paths = [], []
    with torch.no_grad():
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                vector = model.encode_image(preprocess(image).unsqueeze(0).to(device))
                vectors.append((vector / vector.norm(dim=-1, keepdim=True)).cpu())
                valid_paths.append(str(path.resolve()))
            except Exception as error:
                print(f'Skipping {path}: {error}')
    if not vectors:
        raise RuntimeError('No readable images were indexed.')

    index_path = Path(index_path)
    with index_path.open('wb') as handle:
        pickle.dump({
            'model_name': MODEL_NAME,
            'pretrained': PRETRAINED,
            'min_similarity': MIN_SIMILARITY,
            'paths': valid_paths,
            'vectors': torch.cat(vectors, dim=0),
        }, handle)
    return len(valid_paths)


def search(query, top_k=5, index_path=DEFAULT_INDEX_PATH):
    """Return accepted image matches as ``path``, ``score`` dictionaries."""
    if not query or top_k < 1:
        raise ValueError('query must be non-empty and top_k must be positive')
    with Path(index_path).open('rb') as handle:
        index = pickle.load(handle)
    model, _, tokenizer, device = _load_model()
    with torch.no_grad():
        tokens = tokenizer([query]).to(device)
        vector = model.encode_text(tokens)
        vector = vector / vector.norm(dim=-1, keepdim=True)
        scores = (vector.cpu() @ index['vectors'].T).squeeze(0)
    ranked = torch.argsort(scores, descending=True)[:top_k]
    return [
        {'path': index['paths'][position], 'score': round(float(scores[position]), 4)}
        for position in ranked
        if float(scores[position]) >= index.get('min_similarity', MIN_SIMILARITY)
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument('--query', help='Optional text query to run after indexing')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    count = build_index(args.images, args.index)
    print(f'Indexed {count} images into {args.index}')
    if args.query:
        for match in search(args.query, args.top_k, args.index):
            print(f"{match['score']:.4f}  {match['path']}")