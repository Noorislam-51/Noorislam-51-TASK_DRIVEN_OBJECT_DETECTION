# ================= INSTALL =================
# Run this in your terminal first:
#   pip install ultralytics pillow matplotlib

# ================= IMPORTS =================
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import os

# ================= LOAD YOLO =================
yolo_model = YOLO("yolov8n.pt")

# ================= COCO-TASKS DEFINITION =================
coco_tasks = [
    "step on something",          # 1
    "sit comfortably",            # 2
    "place flowers",              # 3
    "get potatoes out of fire",   # 4
    "water plant",                # 5
    "get lemon out of tea",       # 6
    "dig hole",                   # 7
    "open bottle of beer",        # 8
    "open parcel",                # 9
    "serve wine",                 # 10
    "pour sugar",                 # 11
    "smear butter",               # 12
    "extinguish fire",            # 13
    "pound carpet"                # 14
]

# ──────────────────────────────────────────────────────────────────────────────
# [2]  TASK EMBEDDING  →  Action + Quality decomposition
# ──────────────────────────────────────────────────────────────────────────────
TASK_PROFILES = {
    "step on something": {
        "action": "support_body_weight",
        "quality": "stability_height",
        "target": None,
        "desc": "Need a sturdy, elevated surface to stand on"
    },
    "sit comfortably": {
        "action": "support_body_seated",
        "quality": "comfort_softness",
        "target": None,
        "desc": "Need a padded or flat surface at seat height"
    },
    "place flowers": {
        "action": "contain_liquid",
        "quality": "container_openness",
        "target": "flower",
        "desc": "Need a vessel that holds water upright"
    },
    "get potatoes out of fire": {
        "action": "grasp_retrieve_hot",
        "quality": "heat_resistance_grip",
        "target": "fire",
        "desc": "Need a tool that grips and shields from heat"
    },
    "water plant": {
        "action": "pour_liquid",
        "quality": "liquid_capacity",
        "target": "potted plant",
        "desc": "Need a container that pours water with control"
    },
    "get lemon out of tea": {
        "action": "fish_small_object",
        "quality": "reach_precision",
        "target": "cup",
        "desc": "Need a slim elongated tool to reach into a cup"
    },
    "dig hole": {
        "action": "displace_earth",
        "quality": "sharpness_leverage",
        "target": "ground",
        "desc": "Need a rigid pointed or edged implement"
    },
    "open bottle of beer": {
        "action": "lever_pry",
        "quality": "leverage_hardness",
        "target": "bottle",
        "desc": "Need a hard lever-like edge to pop a cap"
    },
    "open parcel": {
        "action": "cut_tear",
        "quality": "sharpness_rigidity",
        "target": "suitcase",
        "desc": "Need a sharp or rigid edge to cut tape/wrapping"
    },
    "serve wine": {
        "action": "pour_controlled",
        "quality": "liquid_capacity_openness",
        "target": None,
        "desc": "Need a vessel that pours gracefully"
    },
    "pour sugar": {
        "action": "pour_granular",
        "quality": "narrow_opening_control",
        "target": None,
        "desc": "Need a small-mouthed vessel for precise pouring"
    },
    "smear butter": {
        "action": "spread_surface",
        "quality": "flat_edge_flexibility",
        "target": "sandwich",
        "desc": "Need a flat flexible blade to smear evenly"
    },
    "extinguish fire": {
        "action": "deliver_liquid_fast",
        "quality": "liquid_capacity_speed",
        "target": "fire",
        "desc": "Need a container that delivers water quickly"
    },
    "pound carpet": {
        "action": "impact_flat_surface",
        "quality": "weight_flat_face",
        "target": "carpet",
        "desc": "Need a heavy flat-faced object to beat dust out"
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# [3]  MULTI-FACTOR SCORING ENGINE
# ──────────────────────────────────────────────────────────────────────────────

ACTION_SCORES = {
    "support_body_weight":        {"chair": 1.8, "bench": 1.6, "table": 1.4, "bed": 1.2, "suitcase": 0.8, "couch": 1.5},
    "support_body_seated":        {"chair": 2.0, "couch": 1.9, "bench": 1.7, "bed": 1.5, "toilet": 1.2},
    "contain_liquid":             {"vase": 2.0, "cup": 1.8, "bowl": 1.6, "bottle": 1.5, "wine glass": 1.4},
    "grasp_retrieve_hot":         {"fork": 1.8, "spoon": 1.6, "knife": 1.5, "scissors": 0.8},
    "pour_liquid":                {"bottle": 2.0, "cup": 1.6, "bowl": 1.3, "vase": 1.0},
    "fish_small_object":          {"fork": 2.0, "spoon": 1.8, "knife": 1.0},
    "displace_earth":             {"knife": 1.0, "fork": 0.8, "scissors": 0.7, "baseball bat": 1.2},
    "lever_pry":                  {"knife": 2.0, "spoon": 1.5, "scissors": 1.2, "fork": 1.0},
    "cut_tear":                   {"knife": 2.0, "scissors": 1.8, "fork": 0.6},
    "pour_controlled":            {"bottle": 1.8, "wine glass": 1.6, "cup": 1.4, "bowl": 1.1},
    "pour_granular":              {"cup": 1.8, "bowl": 1.5, "bottle": 1.7},
    "spread_surface":             {"knife": 2.0, "spoon": 1.4, "fork": 0.6},
    "deliver_liquid_fast":        {"bottle": 2.0, "bowl": 1.6, "cup": 1.3},
    "impact_flat_surface":        {"book": 1.8, "chair": 1.5, "baseball bat": 1.3, "suitcase": 1.0},
}

QUALITY_SCORES = {
    "stability_height":           {"chair": 1.5, "table": 1.8, "bench": 1.4, "bed": 1.0, "suitcase": 1.2, "couch": 1.3},
    "comfort_softness":           {"couch": 2.0, "bed": 1.9, "chair": 1.5, "bench": 1.0},
    "container_openness":         {"vase": 2.0, "bowl": 1.8, "cup": 1.5, "wine glass": 1.4, "bottle": 1.0},
    "heat_resistance_grip":       {"fork": 2.0, "spoon": 1.8, "knife": 1.5},
    "liquid_capacity":            {"bottle": 2.0, "bowl": 1.7, "cup": 1.5, "vase": 1.2},
    "reach_precision":            {"fork": 1.8, "spoon": 1.6, "knife": 1.3},
    "sharpness_leverage":         {"knife": 2.0, "fork": 1.2, "scissors": 1.8},
    "leverage_hardness":          {"knife": 2.0, "spoon": 1.4, "fork": 1.2, "scissors": 1.5},
    "sharpness_rigidity":         {"knife": 2.0, "scissors": 1.9, "fork": 0.8},
    "liquid_capacity_openness":   {"bottle": 1.8, "wine glass": 1.7, "cup": 1.5, "bowl": 1.2},
    "narrow_opening_control":     {"bottle": 2.0, "cup": 1.6, "bowl": 1.0},
    "flat_edge_flexibility":      {"knife": 2.0, "spoon": 1.3},
    "liquid_capacity_speed":      {"bottle": 2.0, "bowl": 1.8, "cup": 1.4},
    "weight_flat_face":           {"book": 2.0, "chair": 1.5, "suitcase": 1.3, "baseball bat": 1.2},
}

TOOL_LIKELIHOOD = {
    "knife":        0.95,
    "fork":         0.90,
    "spoon":        0.88,
    "scissors":     0.85,
    "baseball bat": 0.70,
    "bottle":       0.65,
    "cup":          0.60,
    "bowl":         0.55,
    "book":         0.50,
    "chair":        0.50,
    "bench":        0.45,
    "table":        0.45,
    "couch":        0.40,
    "bed":          0.35,
    "vase":         0.40,
    "wine glass":   0.55,
    "suitcase":     0.30,
    "backpack":     0.30,
    "toilet":       0.25,
    "potted plant": 0.10,
}

PHYSICAL_PROPERTY_MATCH = {
    ("support_body_weight",  "table"):        0.5,
    ("support_body_seated",  "toilet"):       0.8,
    ("contain_liquid",       "vase"):         1.0,
    ("grasp_retrieve_hot",   "fork"):         1.0,
    ("pour_liquid",          "bottle"):       1.0,
    ("fish_small_object",    "fork"):         1.0,
    ("lever_pry",            "knife"):        1.0,
    ("cut_tear",             "scissors"):     1.0,
    ("cut_tear",             "knife"):        0.9,
    ("spread_surface",       "knife"):        1.0,
    ("deliver_liquid_fast",  "bottle"):       0.8,
    ("impact_flat_surface",  "book"):         1.0,
    ("pour_granular",        "bottle"):       0.8,
    ("displace_earth",       "knife"):        0.5,
}

EFFICIENCY_SCORES = {
    ("support_body_weight",  "chair"):        1.0,
    ("support_body_weight",  "table"):        0.9,
    ("support_body_seated",  "couch"):        1.0,
    ("contain_liquid",       "vase"):         1.0,
    ("contain_liquid",       "cup"):          0.9,
    ("grasp_retrieve_hot",   "fork"):         0.9,
    ("pour_liquid",          "bottle"):       1.0,
    ("fish_small_object",    "fork"):         1.0,
    ("fish_small_object",    "spoon"):        0.85,
    ("lever_pry",            "knife"):        0.9,
    ("cut_tear",             "scissors"):     1.0,
    ("cut_tear",             "knife"):        0.85,
    ("spread_surface",       "knife"):        1.0,
    ("deliver_liquid_fast",  "bottle"):       1.0,
    ("impact_flat_surface",  "book"):         0.9,
    ("pour_controlled",      "bottle"):       0.9,
    ("pour_granular",        "bottle"):       0.9,
}

# ──────────────────────────────────────────────────────────────────────────────
# Scoring function
# ──────────────────────────────────────────────────────────────────────────────

def score_object(obj: str, profile: dict) -> dict:
    action  = profile["action"]
    quality = profile["quality"]
    target  = profile.get("target")

    W_ACTION   = 1.0
    W_QUALITY  = 1.0
    W_TOOL     = 0.5
    W_PHYSICAL = 0.8
    W_EFFIC    = 0.7
    W_TARGET   = 2.0

    f_action   = ACTION_SCORES.get(action,  {}).get(obj, 0.0) * W_ACTION
    f_quality  = QUALITY_SCORES.get(quality,{}).get(obj, 0.0) * W_QUALITY
    f_tool     = TOOL_LIKELIHOOD.get(obj, 0.0)                * W_TOOL
    f_physical = PHYSICAL_PROPERTY_MATCH.get((action, obj), 0.0) * W_PHYSICAL
    f_effic    = EFFICIENCY_SCORES.get((action, obj), 0.0)    * W_EFFIC
    penalty    = W_TARGET if (target and target.lower() in obj.lower()) else 0.0
    total      = f_action + f_quality + f_tool + f_physical + f_effic - penalty

    return {
        "action_score":    round(f_action,   3),
        "quality_score":   round(f_quality,  3),
        "tool_likelihood": round(f_tool,     3),
        "physical_match":  round(f_physical, 3),
        "efficiency":      round(f_effic,    3),
        "target_penalty":  round(-penalty,   3),
        "TOTAL":           round(total,      3),
    }

# ──────────────────────────────────────────────────────────────────────────────
# [4]  RANKING + JUGAAD LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def jugaad_fallback(ranked: list, profile: dict) -> str | None:
    if not ranked:
        return None
    best_obj, best_breakdown = ranked[0]
    if best_breakdown["TOTAL"] < 1.0:
        return (
            f"⚠️  Low confidence (score {best_breakdown['TOTAL']:.2f}). "
            f"Consider any rigid container or elongated object nearby as a substitute."
        )
    if best_breakdown["target_penalty"] < 0:
        return (
            f"⚠️  '{best_obj}' appears to be the TARGET of the task. "
            f"Try using '{ranked[1][0]}' instead." if len(ranked) > 1 else
            f"⚠️  '{best_obj}' is the task target — look for an alternative tool."
        )
    return None

# ──────────────────────────────────────────────────────────────────────────────
# GUI: Step 1 — Image picker
# ──────────────────────────────────────────────────────────────────────────────

def pick_image() -> str:
    """Open a native OS file-picker and return the chosen image path."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="📷  Select an Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp"),
            ("All files",   "*.*"),
        ]
    )
    root.destroy()
    return path

# ──────────────────────────────────────────────────────────────────────────────
# GUI: Step 2 — Task selector (listbox + type-to-search)
# ──────────────────────────────────────────────────────────────────────────────

def pick_task() -> str | None:
    """
    GUI window that lets the user:
      • Click any task in the list, OR
      • Type part of a task name and press Enter / click OK
    Returns the matched task string or None if cancelled.
    """
    selected = {"task": None}

    root = tk.Tk()
    root.title("🧠  Select COCO Task")
    root.geometry("430x500")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    # ── Header ──
    tk.Label(root, text="COCO-Tasks", font=("Helvetica", 14, "bold")).pack(pady=(14, 2))
    tk.Label(root, text="Click a task  —  or type below and press Enter / OK",
             font=("Helvetica", 9), fg="#555").pack(pady=(0, 8))

    # ── Listbox + scrollbar ──
    list_frame = tk.Frame(root, bd=1, relief="sunken")
    list_frame.pack(fill="both", expand=True, padx=18, pady=(0, 8))

    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(
        list_frame,
        yscrollcommand=scrollbar.set,
        font=("Courier", 11),
        selectbackground="#4a90e2",
        selectforeground="white",
        activestyle="none",
        height=14,
        cursor="hand2",
    )
    for i, t in enumerate(coco_tasks, 1):
        listbox.insert(tk.END, f"  {i:>2}.  {t}")
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    # ── Type-to-search entry ──
    tk.Label(root, text="Or type task name:", font=("Helvetica", 10)).pack()
    entry_var = tk.StringVar()
    entry = tk.Entry(root, textvariable=entry_var, font=("Helvetica", 11),
                     width=36, relief="solid", bd=1)
    entry.pack(pady=(4, 10))
    entry.focus()

    # Clicking a listbox item fills the entry
    def on_listbox_select(event):
        sel = listbox.curselection()
        if sel:
            # strip the leading index prefix we added
            raw = listbox.get(sel[0]).strip()
            task_name = raw.split(".  ", 1)[-1].strip()
            entry_var.set(task_name)

    listbox.bind("<<ListboxSelect>>", on_listbox_select)

    # Live filter: highlight matching rows as user types
    def on_type(event=None):
        typed = entry_var.get().strip().lower()
        listbox.selection_clear(0, tk.END)
        if not typed:
            return
        for i, t in enumerate(coco_tasks):
            if typed in t.lower():
                listbox.selection_set(i)
                listbox.see(i)
                break

    entry_var.trace_add("write", lambda *_: on_type())

    # ── Confirm logic ──
    def confirm(event=None):
        typed = entry_var.get().strip().lower()
        for t in coco_tasks:
            if typed == t.lower():
                selected["task"] = t
                root.destroy()
                return
        for t in coco_tasks:
            if typed in t.lower():
                selected["task"] = t
                root.destroy()
                return
        messagebox.showwarning(
            "Not found",
            f"'{entry_var.get().strip()}' didn't match any task.\nTry typing a keyword.",
            parent=root,
        )

    entry.bind("<Return>", confirm)

    # ── Buttons ──
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=(0, 14))

    tk.Button(btn_frame, text="✅  OK", command=confirm,
              font=("Helvetica", 11, "bold"),
              bg="#4a90e2", fg="white", padx=22, pady=5,
              relief="flat", cursor="hand2").pack(side="left", padx=6)

    tk.Button(btn_frame, text="✖  Cancel", command=root.destroy,
              font=("Helvetica", 11),
              bg="#e25555", fg="white", padx=14, pady=5,
              relief="flat", cursor="hand2").pack(side="left", padx=6)

    root.mainloop()
    return selected["task"]

# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Step A: Pick image ──
    print("📷 Opening file picker…")
    img_path = pick_image()

    if not img_path:
        print("❌ No image selected. Exiting.")
        sys.exit(1)

    image = Image.open(img_path).convert("RGB")
    print(f"✅ Image loaded : {img_path}")

    plt.imshow(image)
    plt.axis("off")
    plt.title("Input Image")
    plt.tight_layout()
    plt.show()

    # ── Step B: Pick task ──
    print("🧠 Opening task selector…")
    task = pick_task()

    if not task:
        print("❌ No task selected. Exiting.")
        sys.exit(1)

    profile = TASK_PROFILES[task]
    print(f"\n✅ Selected task : {task}")
    print(f"   Action intent : {profile['action']}")
    print(f"   Key quality   : {profile['quality']}")
    print(f"   Description   : {profile['desc']}")

    # ── [1] YOLO Detection ──
    print("\n── [1] Running YOLOv8 detection ──")
    results = yolo_model(image)
    detected_objects = []
    for r in results:
        for c in r.boxes.cls:
            detected_objects.append(yolo_model.names[int(c)])
    detected_objects = list(set(detected_objects))
    print(f"🔍 Detected objects: {detected_objects}")

    if not detected_objects:
        print("❌ No objects detected. Try a clearer image.")
        sys.exit(1)

    # ── [3] Multi-Factor Scoring ──
    print("\n── [3] Multi-Factor Scoring Engine ──")
    scored = {obj: score_object(obj, profile) for obj in detected_objects}

    header = f"{'Object':<16} {'Action':>7} {'Quality':>8} {'Tool':>6} {'Physical':>9} {'Effic':>6} {'Penalty':>8} {'TOTAL':>7}"
    print("\n" + header)
    print("─" * len(header))
    for obj, bd in scored.items():
        print(
            f"{obj:<16} "
            f"{bd['action_score']:>7.3f} "
            f"{bd['quality_score']:>8.3f} "
            f"{bd['tool_likelihood']:>6.3f} "
            f"{bd['physical_match']:>9.3f} "
            f"{bd['efficiency']:>6.3f} "
            f"{bd['target_penalty']:>8.3f} "
            f"{bd['TOTAL']:>7.3f}"
        )

    # ── [4] Ranking ──
    ranked = sorted(scored.items(), key=lambda x: x[1]["TOTAL"], reverse=True)
    print("\n── [4] Ranking (best → worst) ──")
    for rank, (obj, bd) in enumerate(ranked, 1):
        bar = "█" * max(0, int(bd["TOTAL"] * 3))
        print(f"  {rank}. {obj:<16} {bd['TOTAL']:+.3f}  {bar}")

    jugaad = jugaad_fallback(ranked, profile)

    # ── [5] Final Output ──
    print("\n" + "═" * 55)
    print("  [5] FINAL OUTPUT")
    print("═" * 55)
    best_obj, best_bd = ranked[0]
    print(f"\n  Task              : {task}")
    print(f"  Best object       : ✅  {best_obj.upper()}")
    print(f"  Total score       : {best_bd['TOTAL']:.3f}")

    feature_map = {
        "action_score":    "Action suitability",
        "quality_score":   "Physical quality match",
        "tool_likelihood": "Tool-use likelihood",
        "physical_match":  "Physical property alignment",
        "efficiency":      "Task efficiency",
    }
    best_feature_key = max(
        [k for k in best_bd if k not in ("TOTAL", "target_penalty")],
        key=lambda k: best_bd[k]
    )
    print(f"  Best scoring feat : ⭐  {feature_map[best_feature_key]} ({best_bd[best_feature_key]:.3f})")

    if jugaad:
        print(f"\n  Jugaad note       : {jugaad}")

    print("\n" + "═" * 55)


if __name__ == "__main__":
    main()