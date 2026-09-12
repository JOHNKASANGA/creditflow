# ocr/receipt_parser.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import easyocr
import re
import numpy as np
from PIL import Image, ImageOps

reader = easyocr.Reader(['en'], gpu=False)

def load_and_correct(image_path):
    """Shared orientation fix — used by both OCR and the UI preview so they always match."""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")

def parse_receipt(image_path):
    img = load_and_correct(image_path)
    ocr_img = ImageOps.autocontrast(img.convert("L")).convert("RGB")
    img_array = np.array(ocr_img)

    results = reader.readtext(img_array, detail=0)
    text = " ".join(results)

    numbers = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', text)
    numbers = [float(n.replace(",", "")) for n in numbers]

    date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)

    return {
        "raw_text": text,
        "date": date_match.group() if date_match else None,
        "numbers_found": numbers,
    }

if __name__ == "__main__":
    result = parse_receipt(sys.argv[1])
    print(result)