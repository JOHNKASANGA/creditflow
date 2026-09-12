# ocr/receipt_parser.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import easyocr
import re
import numpy as np
from PIL import Image

reader = easyocr.Reader(['en'], gpu=False)

def parse_receipt(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.rotate(90, expand=True)
    img_array = np.array(img)

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