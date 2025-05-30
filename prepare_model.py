import spacy
import requests
from transformers import CLIPProcessor, CLIPModel, MarianMTModel, MarianTokenizer
import torch 

# --- Configuración de modelos ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# --- Descargar keywords de ARASAAC filtrados por "food" ---
def fetch_arasaac_food_keywords(language="es"):
    print("Descargando keywords desde ARASAAC")
    url = f"https://api.arasaac.org/v1/pictograms/{language}/search/food"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        keywords = set()
        for item in data:
            for kw_entry in item.get("keywords", []):
                kw = kw_entry.get("keyword")
                if kw:
                    keywords.add(kw.lower())
        return list(set(keywords))
    else:
        raise Exception(f"No se pudo obtener la lista: {response.status_code}")

LANGUAGE = "es"
nlp = spacy.load("es_core_news_lg")  # Modelo mediano con vectores

arasaac_keywords=fetch_arasaac_food_keywords()

# Preprocesar keywords para spaCy docs
keyword_docs = [(kw, nlp(kw)) for kw in arasaac_keywords if nlp(kw).has_vector]
