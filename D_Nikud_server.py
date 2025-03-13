from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer
from datetime import datetime
import os
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import re
import gzip

# Import your custom modules
from src.models import DNikudModel, ModelConfig
from src.models_utils import predict
from src.running_params import BATCH_SIZE, MAX_LENGTH_SEN
from src.utiles_data import NikudDataset, Nikud, create_missing_folders

app = Flask(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'models/Dnikud_best_model.pth'
COMPRESSED_PARTS_PREFIX = 'models/Dnikud_best_model.gz.part'

def get_logger():
    log_location = os.path.join(Path(__file__).parent, "logging", "server_logs")
    create_missing_folders(log_location)

    log_format = '%(asctime)s %(levelname)-8s Thread_%(thread)-6d ::: %(funcName)s(%(lineno)d) ::: %(message)s'
    logger = logging.getLogger("server")
    logger.setLevel(logging.ERROR)

    file_location = os.path.join(log_location, 'server.log')
    file_handler = RotatingFileHandler(file_location, mode='a', maxBytes=2 * 1024 * 1024, backupCount=20)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    return logger

logger = get_logger()

# Function to extract and reassemble the model file
def extract_model():
    if not os.path.exists(MODEL_PATH):
        compressed_file = MODEL_PATH + ".gz"

        # Reassemble compressed file from parts
        part_files = sorted([f for f in os.listdir("models") if f.startswith("Dnikud_best_model.gz.part")])
        if not part_files:
            raise FileNotFoundError("No split parts found for the model file!")

        with open(compressed_file, "wb") as f_out:
            for part in part_files:
                with open(os.path.join("models", part), "rb") as f_in:
                    f_out.write(f_in.read())

        # Extract the .pth file
        with gzip.open(compressed_file, "rb") as f_in, open(MODEL_PATH, "wb") as f_out:
            f_out.write(f_in.read())

        # Optionally remove the compressed file after extraction
        os.remove(compressed_file)
        print(f"Model extracted successfully: {MODEL_PATH}")

# Ensure model is extracted before loading
extract_model()

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

# Load model and tokenizer
tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")
config_path = os.path.join("models", "config.yml")
config = ModelConfig.load_from_file(config_path)

dnikud_model = DNikudModel(config, len(Nikud.label_2_id["nikud"]), len(Nikud.label_2_id["dagesh"]),
                           len(Nikud.label_2_id["sin"]), device=DEVICE).to(DEVICE)
state_dict_model = dnikud_model.state_dict()
state_dict_model.update(torch.load(MODEL_PATH, map_location=DEVICE))
dnikud_model.load_state_dict(state_dict_model)
dnikud_model.eval()

def apply_manual_fixes(text):
    return re.sub("|".join(before for before, _ in manual_fixes),
                  lambda m: dict(manual_fixes)[re.escape(m.group(0))], text)

@app.route("/predict", methods=["POST"])
def predict_text():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    print(f"Got text: {text}")
    dataset = NikudDataset(tokenizer_tavbert, data_list=[text], logger=logger, max_length=MAX_LENGTH_SEN)
    dataset.prepare_data(name="prediction")
    mtb_prediction_dl = torch.utils.data.DataLoader(dataset.prepered_data, batch_size=BATCH_SIZE)
    all_labels = predict(dnikud_model, mtb_prediction_dl, DEVICE)
    text_data_with_labels = dataset.back_2_text(labels=all_labels)

    # Apply manual fixes using regex
    text_data_with_labels = apply_manual_fixes(text_data_with_labels)

    print(text_data_with_labels)

    return jsonify({"diacritized_text": text_data_with_labels})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
