import os
import requests
import spacy
import torch
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)
# --- CONFIGURACIÓN ---
LANGUAGE = "es"
IMAGE_URL = "https://d266ylt43b08qa.cloudfront.net/naranjo-en-flor-villa-raffo_qohh8iegq4h.webp"
SEARCH_TEXT = "Torta con base de pionono dulce. Helado de oreo bañada en chocolate. Decorada con copetes de dulce de leche y mini galletitas oreo. 10 porciones ya cortadas y separadas en film individuales. Peso aproximado 1.100."
FOLDER = "pictogramas"
MAX_PICTOGRAMS = 15

os.makedirs(FOLDER, exist_ok=True)



# --- Cargar imagen desde URL ---
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


# --- Obtener el mejor keyword (en español) que describe la imagen ---
def get_best_matching_keyword(image_url, keywords_es):
    
    image = load_image_from_url(image_url)
    inputs = clip_processor(text=keywords_es, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)[0]

    best_idx = probs.argmax().item()
    best_keyword_es = keywords_es[best_idx]
    return best_keyword_es.split(" ")[0]

# --- Ajuste de similitud ---
def adjust_similarity(word, similarity, boost_compound=0.05, penalty_diminutive=0.05):
    diminutive_suffixes = ("ito", "ita", "illo", "illa", "itos", "itas", "illos", "illas")
    if word.endswith(diminutive_suffixes):
        similarity -= penalty_diminutive
    if " " in word or "-" in word:
        similarity += boost_compound
    return similarity

# --- Extraer keywords con sinónimos ---
def extract_keywords_with_synonyms(text, keyword_docs, keywords_set, similarity_threshold=0.6):
    doc = nlp(text.lower())
    tokens = [token for token in doc if token.pos_ in ["NOUN", "ADJ", "VERB"] and not token.is_stop and token.has_vector]

    valid_keywords = []
    replaced_words = {}

    for token in tokens:
        token_text = token.text
        if token_text in keywords_set:
            if token_text not in valid_keywords:
                valid_keywords.append(token_text)
        else:
            max_sim = 0
            best_kw = None
            for kw, kw_doc in keyword_docs:
                sim = token.similarity(kw_doc)
                sim = adjust_similarity(kw, sim)
                if sim > max_sim:
                    max_sim = sim
                    best_kw = kw
            if max_sim >= similarity_threshold and best_kw not in valid_keywords:
                valid_keywords.append(best_kw)
                replaced_words[token_text] = best_kw

    print("Palabras clave finales:", valid_keywords)
    print("Reemplazos aplicados:", replaced_words)
    return valid_keywords, replaced_words, doc

# --- Buscar pictogramas en ARASAAC ---
def search_pictograms(keyword):
    url = f"https://api.arasaac.org/api/pictograms/{LANGUAGE}/bestsearch/{keyword}"
    r = requests.get(url)
    if r.status_code == 200:
        results = r.json()
        filtered = [item for item in results if "work" not in item.get("tags", [])]
        return filtered
    return []

# --- Descargar pictograma ---
def download_pictogram(pictogram_id, name):
    img_url = f"https://static.arasaac.org/pictograms/{pictogram_id}/{pictogram_id}_300.png"
    r = requests.get(img_url)
    if r.status_code == 200:
        path = os.path.join(FOLDER, f"{name}_{pictogram_id}.png")
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    return None
def resize_image(img_path, size):
    img = Image.open(img_path)
    return img.resize(size)

# --- Combinar imágenes ---
def combine_images(categorized_paths, output_path):
    from PIL import Image

    canvas_width, canvas_height = 500, 500
    combined = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))

    # Imagen principal (300x300)
    if categorized_paths["main"]:
        main_img = Image.open(categorized_paths["main"]).resize((300, 300), )
        main_y = (canvas_height - 300) // 2
        combined.paste(main_img, (0, main_y), main_img.convert("RGBA"))

    # Imagen secundaria (200x200), a la derecha y centrada verticalmente
    if categorized_paths["secondary"]:
        sec_img = Image.open(categorized_paths["secondary"]).resize((200, 200), )
        sec_y = (canvas_height - 200) // 2
        combined.paste(sec_img, (300, sec_y), sec_img.convert("RGBA"))

    # Imágenes terciarias (hasta 5), 100x100 cada una, en fila abajo
    tertiary_imgs = categorized_paths.get("tertiary", [])[:5]
    n_tertiary = len(tertiary_imgs)
    if n_tertiary > 0:
        total_width = 100 * n_tertiary
        start_x = (canvas_width - total_width) // 2
        y_pos = canvas_height - 100
        for i, path in enumerate(tertiary_imgs):
            img = Image.open(path).resize((100, 100))
            x_pos = start_x + i * 100
            combined.paste(img, (x_pos, y_pos), img.convert("RGBA"))

    combined.save(output_path)


def find_secondary_keyword(main_kw, tokens, keywords_set):
    secondary_kw = None
    main_token = None
    for token in tokens:
        if token.text == main_kw:
            main_token = token
            break
    if not main_token:
        return None

    for token in tokens:
        kw = token.text
        if kw == main_kw or kw not in keywords_set:
            continue
        # 1. Relación de dependencia: ej. "hamburguesa de pollo"
        if token.head == main_token or token in main_token.subtree:
            secondary_kw = kw
            break
        # 2. Substring simple
        if kw in main_kw:
            secondary_kw = kw
            break
    print(secondary_kw)
    return secondary_kw

# --- EJECUCIÓN PRINCIPAL ---

main_keyword = get_best_matching_keyword(IMAGE_URL, arasaac_keywords)
valid_keywords, replacements, doc = extract_keywords_with_synonyms(SEARCH_TEXT, keyword_docs, arasaac_keywords)


if not main_keyword or main_keyword not in arasaac_keywords:
  main_keyword=valid_keywords[0]

if main_keyword and main_keyword not in valid_keywords:
    valid_keywords.append(main_keyword)


secondary_keyword = find_secondary_keyword(main_keyword, doc, arasaac_keywords)
if not secondary_keyword:
  secondary_keyword=[word for word in valid_keywords if word != main_keyword][0]
categorized_paths = {"main": None, "secondary": None, "tertiary": []}
used = set()
for kw in valid_keywords:
    if kw in used:
        continue
    results = search_pictograms(kw)
    print(results)
    if results:
        picto = results[0]
        path = download_pictogram(picto["_id"], kw.replace(" ", "_"))
        if not path:
            continue
        if kw == main_keyword:
            categorized_paths["main"] = path
        elif kw == secondary_keyword:
            categorized_paths["secondary"] = path
        else:
            categorized_paths["tertiary"].append(path)
        used.add(kw)
    if sum([1 for p in categorized_paths.values() if isinstance(p, str)] + [len(categorized_paths['tertiary'])]) >= MAX_PICTOGRAMS:
        break
print(categorized_paths)
if categorized_paths["main"] == None:
  categorized_paths["main"]=categorized_paths["secondary"]
  categorized_paths["secondary"]=None
if any([categorized_paths['main'], categorized_paths['secondary'], categorized_paths['tertiary']]):
    final_path = os.path.join(FOLDER, "resultado.png")
    combine_images(categorized_paths, final_path)
    print(f"Imagen compuesta guardada en: {final_path}")
else:
    print("No se encontraron pictogramas válidos para las palabras clave.")
