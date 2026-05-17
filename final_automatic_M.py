# ═══════════════════════════════════════════════════════════════
#  TASK-AWARE OBJECT DETECTION PIPELINE  (COCO dataset edition)
#
#  Logic  : Code-1 unchanged (tier dict, TF-IDF matcher,
#            rank builder, annotation colours)
#  Images : Code-2 approach (COCO scan, smart search)
#  Flow   : task chosen → silent scan → annotated popup
#            → close window → results printed to terminal
# ═══════════════════════════════════════════════════════════════

# ── INSTALL ───────────────────────────────────────────────────
# !pip install ultralytics pillow matplotlib -q

import math
import re
import os
import random
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════
#  BLOCK 1 — TASK DICTIONARY  (Code 1, untouched)
# ══════════════════════════════════════════════════════════════

TASKS = {
    "step on something": {
        "tier1": ["bench", "chair", "couch", "bed", "dining table"],
        "tier2": ["skateboard", "surfboard", "suitcase", "toilet"]
    },
    "sit comfortably": {
        "tier1": ["couch", "bed", "chair", "bench"],
        "tier2": ["toilet", "suitcase", "backpack", "surfboard"]
    },
    "place flowers": {
        "tier1": ["vase", "bowl", "cup", "wine glass"],
        "tier2": ["bottle", "sink", "dining table"]
    },
    "get potatoes out of fire": {
        "tier1": ["fork", "spoon", "knife"],
        "tier2": ["baseball glove", "tennis racket", "frisbee", "scissors"]
    },
    "water plant": {
        "tier1": ["bottle", "bowl", "cup", "wine glass", "vase"],
        "tier2": ["sink", "umbrella", "backpack"]
    },
    "get lemon out of tea": {
        "tier1": ["fork", "spoon", "knife"],
        "tier2": ["toothbrush", "scissors"]
    },
    "dig hole": {
        "tier1": ["spoon", "knife", "fork"],
        "tier2": ["scissors", "umbrella", "baseball bat", "toothbrush"]
    },
    "open bottle of beer": {
        "tier1": ["knife", "fork", "scissors"],
        "tier2": ["spoon", "remote", "baseball bat", "skateboard"]
    },
    "open parcel": {
        "tier1": ["scissors", "knife"],
        "tier2": ["fork", "baseball bat", "tennis racket"]
    },
    "serve wine": {
        "tier1": ["wine glass", "cup", "bowl"],
        "tier2": ["bottle", "vase", "sink"]
    },
    "pour sugar": {
        "tier1": ["spoon", "cup", "bowl"],
        "tier2": ["bottle", "wine glass", "vase"]
    },
    "smear butter": {
        "tier1": ["knife", "spoon"],
        "tier2": ["fork", "toothbrush", "scissors"]
    },
    "extinguish fire": {
        "tier1": ["fire hydrant", "bottle", "bowl", "cup"],
        "tier2": ["sink", "wine glass", "umbrella"]
    },
    "pound carpet": {
        "tier1": ["baseball bat", "tennis racket"],
        "tier2": ["skateboard", "surfboard", "umbrella", "suitcase", "frisbee"]
    },
}

TASK_NAMES = list(TASKS.keys())

# ══════════════════════════════════════════════════════════════
#  BLOCK 2 — TASK MATCHER  (Code 1, untouched)
#  TF-IDF cosine similarity + overlap rules
# ══════════════════════════════════════════════════════════════

TASK_CORPUS = {
    "step on something": (
        "step on stand on climb on step up get on top boost up elevate "
        "foothold platform reach higher use as step stepping stool "
        "need something to stand on get higher reach something elevated "
        "climb up jump on need a step boost myself lift myself"
    ),
    "sit comfortably": (
        "sit down take a seat have a seat sit comfortably find somewhere to sit "
        "rest my legs rest relax need to sit where can i sit sit for a while "
        "take a rest settle down find seating seat myself lounge comfortable "
        "need a place to sit tired want to rest"
    ),
    "place flowers": (
        "put flowers in arrange flowers keep flowers display flowers store flowers "
        "put bouquet where to put flowers flowers in water flower arrangement "
        "put roses in keep bouquet fresh flowers need water floral arrangement "
        "vase for flowers hold flowers upright place bouquet flower vase"
    ),
    "get potatoes out of fire": (
        "retrieve food from fire take out of fire get food from heat remove from flames "
        "pick up hot food scoop from fire get things out of fire remove from heat "
        "take something from grill fish out from fire grab from fire "
        "remove hot item get veggie from fire potato from heat food from flames "
        "take potato out scoop hot potato retrieve from heat source"
    ),
    "water plant": (
        "water the plant give water to plant irrigate plant pour water on plant "
        "hydrate plant help plant survive water my flower water my tree "
        "pour into plant pot wet the soil water my garden water seedling "
        "irrigate my succulent give water to flower water the garden "
        "pour water onto soil wet the plant hydrate the plant"
    ),
    "get lemon out of tea": (
        "remove lemon from tea fish lemon from cup take lemon out lift lemon out "
        "get fruit out of drink remove slice from drink take out of glass "
        "get something out of a cup fish from hot drink lift from mug "
        "remove lemon slice take citrus out of drink pick lemon from tea"
    ),
    "dig hole": (
        "dig a hole make a hole excavate burrow dig in ground make a pit "
        "break ground dig soil make a trench dig in earth scratch ground "
        "make an opening in ground pierce ground scoop earth dig out "
        "garden digging dig up soil tunnel trench excavate ground"
    ),
    "open bottle of beer": (
        "open beer uncap beer pop the cap remove bottle cap open a bottle "
        "open beer bottle twist cap pry cap off open drink remove lid from bottle "
        "crack open beer open cold one loosen bottle cap uncap bottle "
        "pop open beer remove cap from bottle pry open beer"
    ),
    "open parcel": (
        "open package unbox unwrap parcel cut open box open delivery "
        "cut packaging rip open package open amazon box open wrapped gift "
        "cut tape open sealed box slit open package open envelope "
        "unwrap package tear open box cut open parcel open cardboard box"
    ),
    "serve wine": (
        "pour wine serve a glass of wine fill wine glass pour a drink "
        "pour red wine pour white wine fill glass with wine serve guests wine "
        "wine service pour into glass serve drinks fill up glasses "
        "serve wine to guests fill wine cup pour wine into glass"
    ),
    "pour sugar": (
        "add sugar measure sugar spoon out sugar put sugar in scoop sugar "
        "pour sugar into transfer sugar add sweetener dispense sugar "
        "put some sugar add a spoonful sugar the tea sweeten the drink "
        "measure out sugar add granules spoon sugar into cup"
    ),
    "smear butter": (
        "spread butter apply butter butter bread butter toast coat with butter "
        "spread on bread smear on toast put butter on grease with butter "
        "apply spread smooth butter on slather butter spread on surface "
        "butter the bread coat bread with butter apply butter to toast"
    ),
    "extinguish fire": (
        "put out fire douse fire fight fire kill flames stop burning "
        "extinguish flames quench fire put out flames stop fire "
        "fight the blaze douse the blaze cool fire down stop combustion "
        "smother fire spray water on fire kill the fire put out the blaze"
    ),
    "pound carpet": (
        "beat carpet clean carpet dust carpet hit carpet bang carpet "
        "thwack rug shake out rug beat dust from carpet carpet cleaning "
        "dust out rug smack carpet pound rug beat rug clean "
        "hit the rug beat the carpet remove dust from carpet clean rug"
    ),
}

OVERLAP_RULES = [
    (["fire"],   ["put out","extinguish","douse","stop","quench","spray"],
                 "extinguish fire"),
    (["fire"],   ["food","potato","pick","scoop","hot","grab","retrieve"],
                 "get potatoes out of fire"),
    (["pour"],   ["plant","water","wet","irrigate","soil","garden"],
                 "water plant"),
    (["pour"],   ["wine","glass","drink","serve","guests","red","white"],
                 "serve wine"),
    (["pour"],   ["sugar","sweet","granule","measure","spoon","tea","coffee"],
                 "pour sugar"),
    (["open"],   ["beer","bottle","cap","uncap","cold","drink","lager"],
                 "open bottle of beer"),
    (["open"],   ["box","parcel","package","wrap","delivery","cardboard","gift"],
                 "open parcel"),
    (["plant"],  ["water","wet","irrigate","pour","soil","garden"],
                 "water plant"),
    (["plant"],  ["flower","vase","bouquet","arrange","display","roses"],
                 "place flowers"),
    (["spread"], ["butter","toast","bread","jam","margarine"],
                 "smear butter"),
    (["cut"],    ["box","package","parcel","tape","wrap","cardboard"],
                 "open parcel"),
]

# ── TF-IDF engine (Code 1, untouched) ────────────────────────

def _tokenise(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if len(w) > 1]

def _build_index(corpus_dict):
    tokenised    = {t: _tokenise(txt) for t, txt in corpus_dict.items()}
    vocab        = sorted(set(w for toks in tokenised.values() for w in toks))
    word_to_idx  = {w: i for i, w in enumerate(vocab)}
    V, N         = len(vocab), len(corpus_dict)
    doc_freq     = [0] * V
    for toks in tokenised.values():
        for w in set(toks):
            if w in word_to_idx:
                doc_freq[word_to_idx[w]] += 1
    idf = [math.log((N+1)/(df+1))+1.0 if df > 0 else 0.0 for df in doc_freq]
    vecs = {}
    for task, toks in tokenised.items():
        tf   = [0]*V
        for w in toks:
            if w in word_to_idx: tf[word_to_idx[w]] += 1
        tot  = len(toks) or 1
        vec  = [tf[i]/tot * idf[i] for i in range(V)]
        norm = math.sqrt(sum(x**2 for x in vec)) or 1.0
        vecs[task] = [x/norm for x in vec]
    return word_to_idx, idf, vecs, V

def _vec_query(toks, w2i, idf, V):
    tf   = [0]*V
    for w in toks:
        if w in w2i: tf[w2i[w]] += 1
    tot  = len(toks) or 1
    vec  = [tf[i]/tot * idf[i] for i in range(V)]
    norm = math.sqrt(sum(x**2 for x in vec)) or 1.0
    return [x/norm for x in vec]

def _cosine(a, b):
    return sum(x*y for x,y in zip(a,b))

_W2I, _IDF, _TVECS, _V = _build_index(TASK_CORPUS)

def _match_task(user_input):
    """Returns (task_name, score). Pure Python, no libraries."""
    text = user_input.lower()
    scores = {t: 0.0 for t in TASK_NAMES}
    for triggers, clues, boosted in OVERLAP_RULES:
        if any(t in text for t in triggers) and any(c in text for c in clues):
            scores[boosted] += 3.0
    qvec = _vec_query(_tokenise(user_input), _W2I, _IDF, _V)
    for task in TASK_NAMES:
        scores[task] += _cosine(qvec, _TVECS[task])
    best = max(scores, key=scores.get)
    return best, round(scores[best], 4)

# ══════════════════════════════════════════════════════════════
#  BLOCK 3 — TASK INPUT  (Code 1, untouched)
# ══════════════════════════════════════════════════════════════

def get_task():
    print("\n" + "="*50)
    print("  AVAILABLE TASKS:")
    print("="*50)
    for i, t in enumerate(TASK_NAMES, 1):
        print(f"  {i:2}. {t}")
    print("="*50)
    print("  Enter 1-14  → pick directly")
    print("  Enter 0     → describe in your own words")
    print("="*50)

    while True:
        choice = input("\nYour choice: ").strip()

        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= 14:
                task = TASK_NAMES[n - 1]
                print(f"\n  [OK] Task: '{task}'")
                return task
            elif n == 0:
                user_text = input("  Describe your task: ").strip()
                if not user_text:
                    print("  [!] Empty input, try again")
                    continue
                task, score = _match_task(user_text)
                print(f"\n  [MATCHED] '{task}'  (score={score})")
                return task
            else:
                print("  [!] Enter 0-14")
                continue
        else:
            task, score = _match_task(choice)
            print(f"\n  [MATCHED] '{task}'  (score={score})")
            return task

# ══════════════════════════════════════════════════════════════
#  BLOCK 4 — RANK BUILDER  (Code 1, untouched)
# ══════════════════════════════════════════════════════════════

def build_rank(task):
    rank = {}
    for i, o in enumerate(TASKS[task]["tier1"]):
        rank[o] = i          # tier1: 0 to len-1
    for i, o in enumerate(TASKS[task]["tier2"]):
        rank[o] = 100 + i    # tier2: 100 to 100+len-1
    return rank

# ══════════════════════════════════════════════════════════════
#  BLOCK 5 — COCO IMAGE SELECTION  (Code 2 approach)
#  Only the image sourcing is from Code 2 — nothing else.
# ══════════════════════════════════════════════════════════════

def select_coco_image(task, model, coco_images_dir, max_scan=200):
    """
    Code-2 smart scan: shuffle → scan up to max_scan images →
    pick the one with the highest overlap against this task's
    tier1+tier2 objects.  Silent — no preview, no prompt.

    Returns (pil_image, yolo_results, detected_names, filename).
    """
    relevant_objects = set(TASKS[task]["tier1"] + TASKS[task]["tier2"])
    print(f"\n  Relevant objects for this task: {sorted(relevant_objects)}")

    if not os.path.exists(coco_images_dir):
        raise FileNotFoundError(f"COCO folder not found: {coco_images_dir}")

    image_files = [
        f for f in os.listdir(coco_images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not image_files:
        raise FileNotFoundError("No image files found in the COCO folder.")

    print(f"  Found {len(image_files)} images. "
          f"Scanning up to {max_scan} for task-relevant objects...")

    random.shuffle(image_files)
    candidates    = image_files[:max_scan]
    best_image    = None
    best_path     = None
    best_results  = None
    best_detected = []
    best_overlap  = 0

    for img_file in candidates:
        img_path = os.path.join(coco_images_dir, img_file)
        try:
            img      = Image.open(img_path).convert("RGB")
            results  = model(img, verbose=False)
            detected = list({model.names[int(c)]
                             for r in results for c in r.boxes.cls})
            overlap  = len(set(detected) & relevant_objects)

            if overlap > best_overlap:
                best_overlap  = overlap
                best_image    = img
                best_path     = img_path
                best_results  = results
                best_detected = detected
                found_set     = set(detected) & relevant_objects
                print(f"    Better match: {img_file}  "
                      f"(relevant: {found_set})")

            if best_overlap >= 3:       # good enough — stop early
                break

        except Exception:
            continue

    # ── fallback if nothing relevant found ────────────────────
    if best_image is None:
        fallback_file = image_files[0]
        best_path     = os.path.join(coco_images_dir, fallback_file)
        best_image    = Image.open(best_path).convert("RGB")
        best_results  = model(best_image, verbose=False)
        best_detected = list({model.names[int(c)]
                              for r in best_results for c in r.boxes.cls})
        print(f"  No relevant image found. Using fallback: {fallback_file}")

    img_filename = os.path.basename(best_path)
    print(f"\n  Selected : {img_filename}")
    print(f"  Detected : {best_detected}")
    return best_image, best_results, best_detected, img_filename

# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

# ── Step 1: Load YOLO ─────────────────────────────────────────
print("Loading YOLOv8n...")
model = YOLO("yolov8n.pt")
print("Model ready.\n")

# ── Step 2: Get task (Code 1 logic) ──────────────────────────
task = get_task()
rank = build_rank(task)

# ── Step 3: Select image from COCO silently (Code 2 approach) ─
COCO_IMAGES_DIR = r"C:\Users\acer\OneDrive\Desktop\Nooor\DVCON_Hackathon\coco-tasks\coco\images\val2017"

img_pil, yolo_results, detected_names, img_filename = select_coco_image(
    task, model, COCO_IMAGES_DIR, max_scan=200
)

# ── Step 4: Build 'seen' dict — best-confidence box per class ─
seen = {}
for r in yolo_results:
    for box in r.boxes:
        n = model.names[int(box.cls)]
        c = float(box.conf)
        if n not in seen or c > seen[n][1]:
            seen[n] = (box.xyxy[0].tolist(), c)

# ── Step 5: Match & rank  (Code 1, untouched) ────────────────
matched   = sorted([(rank[n], n, d) for n, d in seen.items() if n in rank])
best_name = matched[0][1] if matched else None

# ── Step 6: Draw annotated boxes  (Code 1, untouched) ────────
img  = img_pil.copy()
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
except Exception:
    font = ImageFont.load_default()

for name, (coords, conf) in seen.items():
    x1, y1, x2, y2 = map(int, coords)
    is_best  = (name == best_name)
    in_tier1 = (name in TASKS[task]["tier1"])
    in_tier2 = (name in TASKS[task]["tier2"])

    if is_best:
        color = "#00C853"
    elif in_tier1:
        color = "#2196F3"
    elif in_tier2:
        color = "#FF9800"
    else:
        color = "#9E9E9E"

    width = 4 if is_best else 2
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    label = f"{'> ' if is_best else ''}{name} {conf:.2f}"
    draw.rectangle([x1, y1-18, x1+len(label)*7, y1], fill=color)
    draw.text((x1+2, y1-16), label, fill="white", font=font)

# ── Step 7: Build title & legend ─────────────────────────────
if best_name:
    tier_str  = "T1" if rank[best_name] < 100 else "T2"
    title_str = (f"Task: {task}  |  Image: {img_filename}  |  "
                 f"Best: {best_name.upper()} [{tier_str}]")
else:
    title_str = f"Task: {task}  |  Image: {img_filename}  |  No match found"

legend = [
    mpatches.Patch(color="#00C853", label="Best match"),
    mpatches.Patch(color="#2196F3", label="Tier 1 (not best)"),
    mpatches.Patch(color="#FF9800", label="Tier 2 fallback"),
    mpatches.Patch(color="#9E9E9E", label="Irrelevant"),
]

# ── Step 8: Show annotated popup — BLOCKING ───────────────────
# Terminal output is held until the user closes this window.
print("\n  [Showing result image — close the window to see terminal output]")
plt.figure(figsize=(10, 7))
plt.imshow(img)
plt.axis("off")
plt.title(title_str, fontsize=12)
plt.legend(handles=legend, loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()          # blocks here until the window is closed

# ── Step 9: Print result to terminal AFTER popup is closed ───
print("\n" + "="*50)
if not matched:
    print(f"  No match found for '{task}'.")
    print(f"  Ideal objects: {TASKS[task]['tier1'][:3]}")
else:
    best_rank  = matched[0][0]
    tier_label = "TIER 1 [Ideal]" if best_rank < 100 else "TIER 2 [Fallback]"
    print(f"  Task   : {task}")
    print(f"  Image  : {img_filename}")
    print(f"  Best   : {best_name.upper()}  [{tier_label}]")
    print(f"\n  All matches (best to worst):")
    for pos, name, (_, conf) in matched:
        tier   = "T1" if pos < 100 else "T2"
        marker = ">" if name == best_name else " "
        print(f"  {marker} [{tier}] {name:<20} conf={conf:.2f}")
print("="*50)

# img.save("result.jpg")
# print("\nResult saved as result.jpg")
print("Done.")