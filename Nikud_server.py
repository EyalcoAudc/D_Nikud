from flask import Flask, request, jsonify
import torch
import os
import re

from NikudModel import DNikudNikudModel, DictaBERTModel

app = Flask(__name__)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Nikud model script")
    parser.add_argument(
        "--model",
        choices=["dnikud", "dicta"],
        default="dicta",
        help="Choose which model to use (default: dicta)"
    )
    return parser.parse_args()

# Load manual fixes
def load_manual_fixes(fixes_file):
    fixes = []
    if os.path.exists(fixes_file):
        with open(fixes_file, "r", encoding="utf-8") as f:
            next(f)  # Skip the first line
            for line in f:
                parts = line.strip().split("|a|")
                if len(parts) == 2:
                    fixes.append((re.escape(parts[0]), parts[1]))
    return fixes

MANUAL_FIXES_FILE = "manual_fixes.txt"
manual_fixes = load_manual_fixes(MANUAL_FIXES_FILE)

def apply_manual_fixes(text):
    return re.sub("|".join(before for before, _ in manual_fixes), lambda m: dict(manual_fixes)[re.escape(m.group(0))], text)

@app.route("/predict", methods=["POST"])
def predict_text():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    print(f"Got text: {text}")

    output = nikud_model.predict(text)

    # Apply manual fixes using regex
    output_fixed = apply_manual_fixes(output)

    print(output_fixed)

    return jsonify({"diacritized_text": output_fixed})


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DNIKUD_MODEL_PATH = 'models/Dnikud/Dnikud_best_model.pth'
DNIKUD_MODEL_CONFIG_PATH = 'models/Dnikud/config.yml'
COMPRESSED_DNIKUD_PARTS_PREFIX = 'models/Dnikud'

DICTA_MODEL_PATH = './models/Dicta'
DICTA_MODEL_CONFIG_PATH = 'models/Dicta/config.json'
COMPRESSED_DICTA_PARTS_PREFIX = './models/Dicta'

args = parse_args()

if args.model == "dnikud":
    nikud_model = DNikudNikudModel(DNIKUD_MODEL_PATH, DEVICE, DNIKUD_MODEL_CONFIG_PATH, COMPRESSED_DNIKUD_PARTS_PREFIX)
elif args.model == "dicta":
    nikud_model = DictaBERTModel(DICTA_MODEL_PATH, DEVICE, DICTA_MODEL_CONFIG_PATH, COMPRESSED_DICTA_PARTS_PREFIX)
else:
    raise ValueError(f"Invalid model: {args.model}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
