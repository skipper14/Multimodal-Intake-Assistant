import torch
import open_clip
from PIL import Image

if torch.cuda.is_available():
    device = 'cuda'   # use the GPU if one is available
else:
    device = 'cpu'    # otherwise fall back to the CPU
# load the pretrained CLIP model (weights download on first run)
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')  # prepares text for the text encoder
model = model.to(device).eval()

TOPICS = ['malaria prevention', 'nutrition', 'maternal and antenatal care',
          'first aid', 'hygiene and sanitation', 'diabetes management']
LABELS = []
for t in TOPICS:
    LABELS.append(f'a health-education poster about {t}')

def classify(image_path):
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')   # open the file as an RGB image
        img = preprocess(image)                         # resize and normalise for CLIP
        img = img.unsqueeze(0).to(device)               # add a batch dimension, move to device
        img_vec = model.encode_image(img)  # image becomes a 512-number vector
        img_vec = img_vec / img_vec.norm(dim=-1, keepdim=True)

        tokens = tokenizer(LABELS).to(device)
        txt_vecs = model.encode_text(tokens)  # text becomes a 512-number vector
        txt_vecs = txt_vecs / txt_vecs.norm(dim=-1, keepdim=True)

        scores = 100.0 * img_vec @ txt_vecs.T   # one similarity score per label
        probs = scores.softmax(dim=-1)          # turn raw scores into confidences that sum to 1
        probs = probs.squeeze(0)                # drop the batch dimension
    pairs = list(zip(TOPICS, probs.tolist()))   # (topic, confidence) pairs
    ranked = sorted(pairs, key=lambda pair: pair[1], reverse=True)   # highest confidence first
    return ranked

if __name__ == '__main__':
    for topic, p in classify('images/poster_07.jpg'):
        print(f'{p:.2%}  {topic}')