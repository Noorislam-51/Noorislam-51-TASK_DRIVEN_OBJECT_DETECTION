from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os

# ── Config ────────────────────────────────────────────────────────────────────
COCO_TASKS = [
    "step on something", "sit comfortably", "place flowers",
    "get potatoes out of fire", "water plant", "get lemon out of tea",
    "dig hole", "open bottle of beer", "open parcel", "serve wine",
    "pour sugar", "smear butter", "extinguish fire", "pound carpet",
]

TASK_PROFILES = {
    "step on something":        ("support_body_weight",   "stability_height",         None,          "Need a sturdy, elevated surface to stand on"),
    "sit comfortably":          ("support_body_seated",   "comfort_softness",         None,          "Need a padded or flat surface at seat height"),
    "place flowers":            ("contain_liquid",         "container_openness",       "flower",      "Need a vessel that holds water upright"),
    "get potatoes out of fire": ("grasp_retrieve_hot",    "heat_resistance_grip",     "fire",        "Need a tool that grips and shields from heat"),
    "water plant":              ("pour_liquid",            "liquid_capacity",          "potted plant","Need a container that pours water with control"),
    "get lemon out of tea":     ("fish_small_object",     "reach_precision",          "cup",         "Need a slim elongated tool to reach into a cup"),
    "dig hole":                 ("displace_earth",         "sharpness_leverage",       "ground",      "Need a rigid pointed or edged implement"),
    "open bottle of beer":      ("lever_pry",              "leverage_hardness",        "bottle",      "Need a hard lever-like edge to pop a cap"),
    "open parcel":              ("cut_tear",               "sharpness_rigidity",       "suitcase",    "Need a sharp or rigid edge to cut tape/wrapping"),
    "serve wine":               ("pour_controlled",        "liquid_capacity_openness", None,          "Need a vessel that pours gracefully"),
    "pour sugar":               ("pour_granular",          "narrow_opening_control",   None,          "Need a small-mouthed vessel for precise pouring"),
    "smear butter":             ("spread_surface",         "flat_edge_flexibility",    "sandwich",    "Need a flat flexible blade to smear evenly"),
    "extinguish fire":          ("deliver_liquid_fast",    "liquid_capacity_speed",    "fire",        "Need a container that delivers water quickly"),
    "pound carpet":             ("impact_flat_surface",    "weight_flat_face",         "carpet",      "Need a heavy flat-faced object to beat dust out"),
}

ACTION_SCORES = {
    "support_body_weight":  {"chair":1.8, "bench":1.6, "table":1.4, "bed":1.2, "suitcase":0.8, "couch":1.5},
    "support_body_seated":  {"chair":2.0, "couch":1.9, "bench":1.7, "bed":1.5, "toilet":1.2},
    "contain_liquid":       {"vase":2.0, "cup":1.8, "bowl":1.6, "bottle":1.5, "wine glass":1.4},
    "grasp_retrieve_hot":   {"fork":1.8, "spoon":1.6, "knife":1.5, "scissors":0.8},
    "pour_liquid":          {"bottle":2.0, "cup":1.6, "bowl":1.3, "vase":1.0},
    "fish_small_object":    {"fork":2.0, "spoon":1.8, "knife":1.0},
    "displace_earth":       {"knife":1.0, "fork":0.8, "scissors":0.7, "baseball bat":1.2},
    "lever_pry":            {"knife":2.0, "spoon":1.5, "scissors":1.2, "fork":1.0},
    "cut_tear":             {"knife":2.0, "scissors":1.8, "fork":0.6},
    "pour_controlled":      {"bottle":1.8, "wine glass":1.6, "cup":1.4, "bowl":1.1},
    "pour_granular":        {"cup":1.8, "bowl":1.5, "bottle":1.7},
    "spread_surface":       {"knife":2.0, "spoon":1.4, "fork":0.6},
    "deliver_liquid_fast":  {"bottle":2.0, "bowl":1.6, "cup":1.3},
    "impact_flat_surface":  {"book":1.8, "chair":1.5, "baseball bat":1.3, "suitcase":1.0},
}

QUALITY_SCORES = {
    "stability_height":        {"chair":1.5, "table":1.8, "bench":1.4, "bed":1.0, "suitcase":1.2, "couch":1.3},
    "comfort_softness":        {"couch":2.0, "bed":1.9, "chair":1.5, "bench":1.0},
    "container_openness":      {"vase":2.0, "bowl":1.8, "cup":1.5, "wine glass":1.4, "bottle":1.0},
    "heat_resistance_grip":    {"fork":2.0, "spoon":1.8, "knife":1.5},
    "liquid_capacity":         {"bottle":2.0, "bowl":1.7, "cup":1.5, "vase":1.2},
    "reach_precision":         {"fork":1.8, "spoon":1.6, "knife":1.3},
    "sharpness_leverage":      {"knife":2.0, "fork":1.2, "scissors":1.8},
    "leverage_hardness":       {"knife":2.0, "spoon":1.4, "fork":1.2, "scissors":1.5},
    "sharpness_rigidity":      {"knife":2.0, "scissors":1.9, "fork":0.8},
    "liquid_capacity_openness":{"bottle":1.8, "wine glass":1.7, "cup":1.5, "bowl":1.2},
    "narrow_opening_control":  {"bottle":2.0, "cup":1.6, "bowl":1.0},
    "flat_edge_flexibility":   {"knife":2.0, "spoon":1.3},
    "liquid_capacity_speed":   {"bottle":2.0, "bowl":1.8, "cup":1.4},
    "weight_flat_face":        {"book":2.0, "chair":1.5, "suitcase":1.3, "baseball bat":1.2},
}

TOOL_LIKELIHOOD = {
    "knife":0.95, "fork":0.90, "spoon":0.88, "scissors":0.85,
    "baseball bat":0.70, "bottle":0.65, "cup":0.60, "bowl":0.55,
    "book":0.50, "chair":0.50, "bench":0.45, "table":0.45,
    "couch":0.40, "bed":0.35, "vase":0.40, "wine glass":0.55,
    "suitcase":0.30, "backpack":0.30, "toilet":0.25, "potted plant":0.10,
}

PHYSICAL_PROPERTY_MATCH = {
    ("support_body_weight", "table"):  0.5,
    ("support_body_seated", "toilet"): 0.8,
    ("contain_liquid", "vase"):        1.0,
    ("grasp_retrieve_hot", "fork"):    1.0,
    ("pour_liquid", "bottle"):         1.0,
    ("fish_small_object", "fork"):     1.0,
    ("lever_pry", "knife"):            1.0,
    ("cut_tear", "scissors"):          1.0,
    ("cut_tear", "knife"):             0.9,
    ("spread_surface", "knife"):       1.0,
    ("deliver_liquid_fast", "bottle"): 0.8,
    ("impact_flat_surface", "book"):   1.0,
    ("pour_granular", "bottle"):       0.8,
    ("displace_earth", "knife"):       0.5,
}

EFFICIENCY_SCORES = {
    ("support_body_weight", "chair"):  1.0,
    ("support_body_weight", "table"):  0.9,
    ("support_body_seated", "couch"):  1.0,
    ("contain_liquid", "vase"):        1.0,
    ("contain_liquid", "cup"):         0.9,
    ("grasp_retrieve_hot", "fork"):    0.9,
    ("pour_liquid", "bottle"):         1.0,
    ("fish_small_object", "fork"):     1.0,
    ("fish_small_object", "spoon"):    0.85,
    ("lever_pry", "knife"):            0.9,
    ("cut_tear", "scissors"):          1.0,
    ("cut_tear", "knife"):             0.85,
    ("spread_surface", "knife"):       1.0,
    ("deliver_liquid_fast", "bottle"): 1.0,
    ("impact_flat_surface", "book"):   0.9,
    ("pour_controlled", "bottle"):     0.9,
    ("pour_granular", "bottle"):       0.9,
}

W = {"action": 1.0, "quality": 1.0, "tool": 0.5, "physical": 0.8, "effic": 0.7, "target": 2.0}

FEATURE_LABELS = {
    "action_score":    "Action suitability",
    "quality_score":   "Physical quality match",
    "tool_likelihood": "Tool-use likelihood",
    "physical_match":  "Physical property alignment",
    "efficiency":      "Task efficiency",
}

# ── Core Functions ─────────────────────────────────────────────────────────────
def score_object(obj: str, action: str, quality: str, target: str | None) -> dict:
    f_action   = ACTION_SCORES.get(action, {}).get(obj, 0.0)   * W["action"]
    f_quality  = QUALITY_SCORES.get(quality, {}).get(obj, 0.0) * W["quality"]
    f_tool     = TOOL_LIKELIHOOD.get(obj, 0.0)                 * W["tool"]
    f_physical = PHYSICAL_PROPERTY_MATCH.get((action, obj), 0.0) * W["physical"]
    f_effic    = EFFICIENCY_SCORES.get((action, obj), 0.0)     * W["effic"]
    penalty    = W["target"] if (target and target.lower() in obj.lower()) else 0.0
    total      = f_action + f_quality + f_tool + f_physical + f_effic - penalty
    return {
        "action_score":    round(f_action, 3),
        "quality_score":   round(f_quality, 3),
        "tool_likelihood": round(f_tool, 3),
        "physical_match":  round(f_physical, 3),
        "efficiency":      round(f_effic, 3),
        "target_penalty":  round(-penalty, 3),
        "TOTAL":           round(total, 3),
    }


def jugaad_fallback(ranked: list, target: str | None) -> str | None:
    if not ranked:
        return None
    best_obj, best_bd = ranked[0]
    if best_bd["TOTAL"] < 1.0:
        return f"⚠️ Low confidence ({best_bd['TOTAL']:.2f}). Consider any rigid container or elongated object nearby."
    if best_bd["target_penalty"] < 0:
        alt = f"Try '{ranked[1][0]}' instead." if len(ranked) > 1 else "Look for an alternative tool."
        return f"⚠️ '{best_obj}' is the task target. {alt}"
    return None


def print_table(scored: dict) -> None:
    header = f"{'Object':<16} {'Action':>7} {'Quality':>8} {'Tool':>6} {'Physical':>9} {'Effic':>6} {'Penalty':>8} {'TOTAL':>7}"
    print("\n" + header + "\n" + "─" * len(header))
    for obj, bd in scored.items():
        print(f"{obj:<16} {bd['action_score']:>7.3f} {bd['quality_score']:>8.3f} "
              f"{bd['tool_likelihood']:>6.3f} {bd['physical_match']:>9.3f} "
              f"{bd['efficiency']:>6.3f} {bd['target_penalty']:>8.3f} {bd['TOTAL']:>7.3f}")


def get_relevant_objects(action: str, quality: str) -> set:
    """Return all objects that score > 0 for this task's action or quality profile."""
    relevant = set()
    relevant.update(ACTION_SCORES.get(action, {}).keys())
    relevant.update(QUALITY_SCORES.get(quality, {}).keys())
    return relevant


def show_image_with_boxes(image: Image.Image, results, model, img_filename: str, task: str) -> None:
    """Display the selected image with YOLO bounding boxes."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(image)

    colors = plt.cm.get_cmap("tab20", 20)
    label_color_map = {}
    color_idx = 0

    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            label = model.names[int(cls)]
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1

            if label not in label_color_map:
                label_color_map[label] = colors(color_idx % 20)
                color_idx += 1
            color = label_color_map[label]

            rect = patches.Rectangle((x1, y1), w, h,
                                      linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{label} {conf:.2f}",
                    color="white", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.85))

    ax.set_title(f"📷 {img_filename}  |  Task: {task}", fontsize=11, pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.show(block=False)   # non-blocking so terminal stays alive
    plt.pause(0.5)
    print("\n(Close the image window or press any key in the terminal to continue...)")


# ── Main Pipeline ──────────────────────────────────────────────────────────────

# ── STEP 1: Select task FIRST ──────────────────────────────────────────────────
print("\nAvailable COCO-Tasks:")
for i, t in enumerate(COCO_TASKS, 1):
    print(f"{i:2d}. {t}")

task_idx = int(input("\n🧠 Enter task number (1-14): ")) - 1
task     = COCO_TASKS[task_idx]
action, quality, target, desc = TASK_PROFILES[task]

print(f"\n✅ Task    : {task}")
print(f"   Action  : {action}")
print(f"   Quality : {quality}")
print(f"   Hint    : {desc}")

# ── STEP 2: Find objects relevant to this task ─────────────────────────────────
relevant_objects = get_relevant_objects(action, quality)
print(f"\n🎯 Relevant objects for this task: {sorted(relevant_objects)}")

# ── STEP 3: Validate image directory ──────────────────────────────────────────
coco_images_dir = r"C:\Users\acer\OneDrive\Desktop\Nooor\DVCON_Hackathon\coco-tasks\coco\images\val2017"

if not os.path.exists(coco_images_dir):
    print(f"❌ Folder not found: {coco_images_dir}")
    exit(1)

image_files = [f for f in os.listdir(coco_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not image_files:
    print("❌ No image files found in the folder.")
    exit(1)

print(f"✅ Found {len(image_files)} images in val2017.")

# ── STEP 4: Load YOLO once ─────────────────────────────────────────────────────
print("\n── Loading YOLOv8 ──")
model = YOLO("yolov8n.pt")

# ── STEP 5: Smart image search — find one that contains a relevant object ──────
print("🔍 Scanning images for task-relevant objects (checking up to 200)...")

random.shuffle(image_files)
candidates   = image_files[:200]   # scan at most 200 to stay fast
best_image   = None
best_path    = None
best_results = None
best_detected = []
best_overlap  = 0

for img_file in candidates:
    img_path = os.path.join(coco_images_dir, img_file)
    try:
        img = Image.open(img_path).convert("RGB")
        results = model(img, verbose=False)
        detected = list({model.names[int(c)] for r in results for c in r.boxes.cls})
        overlap  = len(set(detected) & relevant_objects)

        if overlap > best_overlap:
            best_overlap  = overlap
            best_image    = img
            best_path     = img_path
            best_results  = results
            best_detected = detected
            print(f"   ✅ Better match found: {img_file}  (relevant objects: {set(detected) & relevant_objects})")

        if best_overlap >= 3:           # good enough — stop early
            break

    except Exception as e:
        continue

# Fallback: use best found even if overlap == 0
if best_image is None:
    fallback_file = image_files[0]
    best_path     = os.path.join(coco_images_dir, fallback_file)
    best_image    = Image.open(best_path).convert("RGB")
    best_results  = model(best_image, verbose=False)
    best_detected = list({model.names[int(c)] for r in best_results for c in r.boxes.cls})
    print(f"⚠️  No relevant image found. Using fallback: {fallback_file}")

img_filename = os.path.basename(best_path)
print(f"\n📷 Selected image : {img_filename}")
print(f"🔍 Detected objects: {best_detected}")

# ── STEP 6: Show the image so user can see what was selected ───────────────────
show_image_with_boxes(best_image, best_results, model, img_filename, task)

# ── STEP 7: Confirm or skip ────────────────────────────────────────────────────
print("\nThe image above has been selected based on your task.")
print("Press ENTER to run scoring on it, or type 'skip' to pick another random one: ", end="")
user_input = input().strip().lower()

if user_input == "skip":
    # Pick a purely random image from the remaining candidates
    fallback_file = random.choice(image_files)
    best_path     = os.path.join(coco_images_dir, fallback_file)
    best_image    = Image.open(best_path).convert("RGB")
    best_results  = model(best_image, verbose=False)
    best_detected = list({model.names[int(c)] for r in best_results for c in r.boxes.cls})
    img_filename  = fallback_file
    print(f"\n📷 New random image : {img_filename}")
    print(f"🔍 Detected objects : {best_detected}")
    show_image_with_boxes(best_image, best_results, model, img_filename, task)
    input("\nPress ENTER to continue to scoring...")

# ── STEP 8: Scoring & Ranking ──────────────────────────────────────────────────
if not best_detected:
    print("❌ No objects detected in selected image.")
    exit(1)

scored = {obj: score_object(obj, action, quality, target) for obj in best_detected}
print_table(scored)

ranked = sorted(scored.items(), key=lambda x: x[1]["TOTAL"], reverse=True)
print("\n── Ranking ──")
for rank, (obj, bd) in enumerate(ranked, 1):
    bar = "█" * max(0, int(bd["TOTAL"] * 3))
    print(f" {rank}. {obj:<16} {bd['TOTAL']:+.3f} {bar}")

# ── STEP 9: Final output ───────────────────────────────────────────────────────
best_obj, best_bd = ranked[0]
best_feat_key = max(
    (k for k in best_bd if k not in ("TOTAL", "target_penalty")),
    key=lambda k: best_bd[k]
)
jugaad = jugaad_fallback(ranked, target)

print("\n" + "═" * 55)
print(f" Task         : {task}")
print(f" Image        : {img_filename}")
print(f" Best object  : ✅ {best_obj.upper()} (score: {best_bd['TOTAL']:.3f})")
print(f" Top feature  : ⭐ {FEATURE_LABELS[best_feat_key]} ({best_bd[best_feat_key]:.3f})")
if jugaad:
    print(f" Jugaad note  : {jugaad}")
print("═" * 55)

plt.show()   # keep any open figure alive until user closes it1python