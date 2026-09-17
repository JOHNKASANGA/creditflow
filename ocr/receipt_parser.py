import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import re
from PIL import Image, ImageOps
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
load_dotenv()

AZURE_VISION_ENDPOINT = os.environ.get("AZURE_VISION_ENDPOINT")
AZURE_VISION_KEY = os.environ.get("AZURE_VISION_KEY")

client = ImageAnalysisClient(
    endpoint=AZURE_VISION_ENDPOINT,
    credential=AzureKeyCredential(AZURE_VISION_KEY),
)

def load_and_correct(image_path):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")

def parse_receipt(image_path):
    img = load_and_correct(image_path)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    result = client.analyze(
        image_data=image_bytes,
        visual_features=[VisualFeatures.READ],
    )

    lines = []
    if result.read is not None:
        for block in result.read.blocks:
            for line in block.lines:
                lines.append(line.text)
    text = "\n".join(lines)

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