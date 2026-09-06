"""Create ambiguous and out-of-scope fixtures for captioning tests."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUTPUT_DIR = Path(__file__).resolve().parent / 'caption_tests'


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ambiguous = Image.new('RGB', (768, 512), '#9ca3af')
    ambiguous = ambiguous.filter(ImageFilter.GaussianBlur(18))
    ambiguous.save(OUTPUT_DIR / 'ambiguous_blurred.png')

    out_of_scope = Image.new('RGB', (768, 512), '#1d4ed8')
    draw = ImageDraw.Draw(out_of_scope)
    draw.ellipse((190, 100, 570, 480), fill='#facc15')
    draw.text((245, 235), 'LANDSCAPE', fill='#111827')
    out_of_scope.save(OUTPUT_DIR / 'out_of_scope.png')
    print(f'Created caption fixtures in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()