# Pictogram Generator with ARASAAC and CLIP

This script processes descriptive text and optionally an image to generate pictograms using the ARASAAC API. The pictograms are categorized as main, secondary, or tertiary based on relevance and are composed into a single image.

## Prerequisites

- Python 3.8+
- Install the required libraries:
```bash
pip install requests pillow spacy
python -m spacy download es_core_news_lg
pip install -U sentence-transformers transformers
```

## Model Loading

```
python prepare_model.py
```

## Script Execution

- Set the `SEARCH_TEXT` and optionally the `IMAGE_URL` to describe the dish.
- The script will extract keywords, identify the primary element visually (if image provided), and fetch pictograms from ARASAAC.
- Pictograms are categorized and combined into a final image.
```  
  python pictogram_generator.py
```
This will:

Analyze the input text and/or image

Extract keywords (e.g., "hamburguesa", "queso", "papas fritas")

Retrieve matching pictograms from ARASAAC

Compose a 400x400 image with layout:

Main item (300x300) on the left

Secondary (100x100) on the right

Tertiary items (up to 4, 100x100) in a row at the bottom

## Disclaimer

The pictographic symbols used are the property of the Government of Aragón and were created by Sergio Palao for ARASAAC (http://www.arasaac.org), which distributes them under a Creative Commons BY-NC-SA license.
