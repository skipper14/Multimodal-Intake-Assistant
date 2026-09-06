"""Create a small, reproducible health-poster corpus for the CLIP lab."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).resolve().parent / 'images'
FONT_PATH = '/System/Library/Fonts/Supplemental/Arial.ttf'
POSTERS = [
    ('malaria prevention', '#0b3d2e'),
    ('malaria symptoms', '#14532d'),
    ('nutrition and balanced diet', '#365314'),
    ('child nutrition', '#713f12'),
    ('maternal antenatal care', '#9f1239'),
    ('newborn care', '#831843'),
    ('first aid and emergency care', '#991b1b'),
    ('wound care', '#b91c1c'),
    ('hygiene and hand washing', '#075985'),
    ('safe drinking water', '#0369a1'),
    ('diabetes management', '#3730a3'),
    ('blood pressure screening', '#4338ca'),
    ('vaccination schedule', '#6b21a8'),
    ('mental health support', '#7c2d12'),
    ('sexual and reproductive health', '#9a3412'),
]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    heading_font = ImageFont.truetype(FONT_PATH, 28)
    topic_font = ImageFont.truetype(FONT_PATH, 46)
    footer_font = ImageFont.truetype(FONT_PATH, 20)
    for number, (topic, colour) in enumerate(POSTERS, start=1):
        image = Image.new('RGB', (768, 512), colour)
        draw = ImageDraw.Draw(image)
        draw.rectangle((36, 36, 732, 476), outline='#fef3c7', width=4)
        draw.text((72, 90), 'AFYAPLUS HEALTH GUIDE', fill='#fef3c7', font=heading_font)
        words = topic.upper().split()
        lines, current = [], ''
        for word in words:
            candidate = f'{current} {word}'.strip()
            if draw.textlength(candidate, font=topic_font) <= 620:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        draw.multiline_text((72, 205), '\n'.join(lines), fill='white', font=topic_font, spacing=10)
        draw.text((72, 425), 'Seek qualified care for urgent symptoms.', fill='#fef3c7', font=footer_font)
        image.save(OUTPUT_DIR / f'poster_{number:02d}.png')
    print(f'Created {len(POSTERS)} images in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()