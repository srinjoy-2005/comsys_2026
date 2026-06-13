import os
import cv2
import numpy as np
import random
import json
import regex  # <-- CRITICAL: Required for Grapheme Cluster mapping
from PIL import Image, ImageDraw, ImageFont
import requests

DISCORD_WEBHOOK_URL = r"[REDACTED]"

def notify(message):
    print(message)
    if not DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL == "[REDACTED]":
        return
    try:
        requests.post(
            DISCORD_WEBHOOK_URL,
            data=json.dumps({"content": message}),
            headers={"Content-Type": "application/json"},
            timeout=10
        )
    except Exception as e:
        print(f"Notification failed: {e}")

notify("🚀 YOLO + Classifier Dataset generation started...")

# --- 1. CONFIGURATION & TARGET PIPELINE ---
NUM_TEST_IMAGES = 10000 
PAGE_WIDTH, PAGE_HEIGHT = 800, 1131
OUTPUT_DIR = "dataset_production_test1"
CLASSIFIER_DIR = "classifier_dataset"

# Safety cap to prevent Kaggle Inode crashing
MAX_CROPS_PER_CLASS = 1000 
crop_counts = {}

FONT_PATHS = {
    "english_cursive": [
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/Caveat-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/CedarvilleCursive-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/DancingScript-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/GochiHand-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/GreatVibes-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/HomemadeApple-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/IndieFlower-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/LaBelleAurore-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/NothingYouCouldDo-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/Pacifico-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/PlaywriteGBJ-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/PlaywriteUSTrad-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/ReenieBeanie-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/ShadowsIntoLight-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/Zeyada-Regular.ttf",
    ],
    "english_plain": [
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/ComingSoon-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/Lato-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/Montserrat-Italic.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/eng/RobotoFlex-VariableFont_GRAD,XOPQ,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght.ttf",
    ],
    "bengali": [
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/bengali/Mina-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/bengali/BalooDa2-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/bengali/Mina-Regular.ttf",
    ],
    "hindi": [
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/hindi/Amita-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/hindi/Hind-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/hindi/Kalam-Regular.ttf",
        r"/kaggle/input/datasets/thenamelessmonster/fonts-syntheticdatageneration/fonts/hindi/Poppins-Regular.ttf",
    ]
}

TEXT_FILES = {
    "english": "/kaggle/input/datasets/thenamelessmonster/text-corpus/english_corpus.txt",
    "bengali": "/kaggle/input/datasets/thenamelessmonster/text-corpus/bengali_corpus.txt",
    "hindi": "/kaggle/input/datasets/thenamelessmonster/text-corpus/hindi_corpus.txt"
}

for d in [OUTPUT_DIR, CLASSIFIER_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# --- 2. LOCAL CORPUS STREAMING LOADER ---
def load_local_corpus_tokens(lang_key):
    file_path = TEXT_FILES[lang_key]
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tokens = content.split()
            if len(tokens) > 10:
                return tokens
    fallbacks = {
        "english": ["Machine", "learning", "gradient", "descent", "backpropagation", "loss", "matrix", "vector"],
        "bengali": ["কলকাতা", "পশ্চিমবঙ্গ", "বিজ্ঞান", "প্রযুক্তি", "বিশ্ববিদ্যালয়", "শিক্ষা", "গবেষণা"],
        "hindi": ["भारत", "संस्कृति", "भाषा", "सॉफ्टवेयर", "अभियांत्रिकी", "गणित", "विज्ञान", "विकास"]
    }
    return fallbacks[lang_key]

corpus_tokens = {lang: load_local_corpus_tokens(lang) for lang in ["english", "bengali", "hindi"]}

# --- 2.5 LOAD CORPUS STATE ---
STATE_FILE = "corpus_state.json"
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "r") as f:
        corpus_state = json.load(f)
else:
    corpus_state = {"english": 0, "bengali": 0, "hindi": 0}

def save_corpus_state():
    with open(STATE_FILE, "w") as f:
        json.dump(corpus_state, f)

# --- 3. COORDINATE CONVERSION MATH ---
def convert_to_yolo_format(box, img_w, img_h):
    _, x, y, w, h = box
    center_x = (x + (w / 2.0)) / img_w
    center_y = (y + (h / 2.0)) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    return f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"

# --- 4. STREAM GENERATION CORE ENGINE ---
def generate_synthetic_page(image_id, execution_category):
    if execution_category in ["english_cursive", "english_plain"]:
        lang_source = "english"
    else:
        lang_source = execution_category

    tokens = corpus_tokens[lang_source]
    font_pool = FONT_PATHS[execution_category]
    selected_font_path = random.choice(font_pool)

    font_size = random.randint(32, 46) 
    line_spacing_step = font_size + random.randint(8, 18)
    
    font = ImageFont.truetype(selected_font_path, font_size)
    page_canvas = Image.new('RGB', (PAGE_WIDTH, PAGE_HEIGHT), color='white')
    draw = ImageDraw.Draw(page_canvas)

    cursor_x, cursor_y = random.randint(40, 60), random.randint(40, 60)
    bounding_boxes = []

    sample_size = min(len(tokens), 300)
    start_index = corpus_state[lang_source]
    
    if start_index + sample_size > len(tokens):
        page_words = tokens[start_index:len(tokens)]
        remaining = sample_size - len(page_words)
        page_words += tokens[0:remaining]
        corpus_state[lang_source] = remaining
    else:
        page_words = tokens[start_index:start_index + sample_size]
        corpus_state[lang_source] += sample_size

    ink_color = random.choice([
        (10, 10, 10), (25, 25, 35), (15, 30, 85), (30, 50, 110)
    ])

    for word in page_words:
        cursor_y += random.randint(-2, 2)
        word_width = int(draw.textlength(word, font=font))

        if cursor_x + word_width > (PAGE_WIDTH - 50):
            cursor_x = random.randint(40, 60)
            cursor_y += line_spacing_step
            if cursor_y > (PAGE_HEIGHT - 80):
                break

        grapheme_clusters = regex.findall(r'\X', word)

        if execution_category == "english_cursive":
            overlap_tightness = random.randint(1, 3)
        elif execution_category in ["bengali", "hindi"]:
            overlap_tightness = random.randint(2, 4)
        else: 
            overlap_tightness = 0
            
        for cluster in grapheme_clusters:
            bbox = draw.textbbox((cursor_x, cursor_y), cluster, font=font)
            l, t, r, b = bbox
            
            draw.text((cursor_x, cursor_y), cluster, font=font, fill=ink_color)
            
            if cluster.strip():
                # 1. Save YOLO Box
                bounding_boxes.append((cluster, l, t, r - l, b - t))
                
                # 2. Extract and Save Classifier Crop
                # Use UTF-8 hex to avoid OS filesystem forbidden character crashes
                unicode_hex = cluster.encode('utf-8').hex()
                
                # Initialize counter for this character
                if unicode_hex not in crop_counts:
                    crop_counts[unicode_hex] = 0
                
                # Only save crop if we haven't hit the Kaggle inode safety cap
                if crop_counts[unicode_hex] < MAX_CROPS_PER_CLASS:
                    pad = 2
                    crop_l, crop_t = max(0, l - pad), max(0, t - pad)
                    crop_r, crop_b = min(PAGE_WIDTH, r + pad), min(PAGE_HEIGHT, b + pad)
                    
                    if crop_r - crop_l > 3 and crop_b - crop_t > 3:
                        char_crop = page_canvas.crop((crop_l, crop_t, crop_r, crop_b))
                        save_dir = os.path.join(CLASSIFIER_DIR, unicode_hex)
                        os.makedirs(save_dir, exist_ok=True)
                        
                        crop_name = f"{image_id}_{cursor_x}_{cursor_y}.jpg"
                        char_crop.save(os.path.join(save_dir, crop_name))
                        crop_counts[unicode_hex] += 1
                
            cluster_width = int(draw.textlength(cluster, font=font))
            cursor_x += (cluster_width - overlap_tightness)

        cursor_x += int(draw.textlength(" ", font=font)) + random.randint(4, 12)

    img_filepath = os.path.join(OUTPUT_DIR, f"synthetic_{image_id}.jpg")
    txt_filepath = os.path.join(OUTPUT_DIR, f"synthetic_{image_id}.txt")

    page_canvas.save(img_filepath)
    with open(txt_filepath, "w", encoding="utf-8") as f:
        for box in bounding_boxes:
            f.write(convert_to_yolo_format(box, PAGE_WIDTH, PAGE_HEIGHT) + "\n")

# --- 6. EXECUTION DISPATCH LOOP ---
categories = (
    ["english_cursive"] * 60 + 
    ["english_plain"] * 15 + 
    ["bengali"] * 13 + 
    ["hindi"] * 12
)
random.shuffle(categories)

print(f"Starting sample generation inside context path: '{OUTPUT_DIR}'...")

for iteration_id in range(NUM_TEST_IMAGES):
    selected_style = random.choice(categories)
    generate_synthetic_page(iteration_id, selected_style)

    if (iteration_id + 1) % 1000 == 0:
        notify(
            f"✅ Generated {iteration_id + 1:,}/{NUM_TEST_IMAGES:,} images "
            f"({100 * (iteration_id + 1) / NUM_TEST_IMAGES:.1f}% complete)"
        )

save_corpus_state()

notify(f"🎉 Dataset generation finished! Generated {NUM_TEST_IMAGES:,} images and extracted balanced character crops.")
print(f"[SUCCESS] Datasets generated. Check '{OUTPUT_DIR}' and '{CLASSIFIER_DIR}'.")