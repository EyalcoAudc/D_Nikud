import gzip
import os
import tarfile

import torch
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from transformers import AutoModel, AutoTokenizer
from src.models import DNikudModel, ModelConfig
import shutil

# Import your custom modules
from src.models_utils import predict
from src.running_params import BATCH_SIZE, MAX_LENGTH_SEN
from src.utiles_data import NikudDataset, Nikud, create_missing_folders

class NikudModel:
    def __init__(self, model_path: str, device: str, config_path: str, compressed_path: str):
        self.model_path = model_path
        self.device = device
        self.config_path = config_path
        self.compressed_path = compressed_path

        # Ensure model is extracted before loading
        if compressed_path:
            self.extract_split_tar_gz(compressed_path)

    def predict_multiple(self, sentences: list[str]) -> list[str]:
        results = []
        for sentence in sentences:
            results.append(self.predict(sentence))
        return results

    def predict(self, sentence: str) -> str:
        raise NotImplementedError

    # Function to extract and reassemble the model file
    def extract_split_tar_gz(self, dir_path: str):
        dir_path = Path(dir_path)

        # Find all part files like *.gz.partaa, *.gz.partab, etc.
        part_files = sorted(dir_path.glob("*.gz.part*"))
        if not part_files:
            print("No split .tar.gz part files found.")
            return

        # Group part files by base name (before .gz.part*)
        grouped_files = {}
        for part_file in part_files:
            base_name = part_file.name.split('.gz.part')[0]
            grouped_files.setdefault(base_name, []).append(part_file)

        for base_name, parts in grouped_files.items():
            tar_path = dir_path / f"{base_name}.tar.gz"

            # Check if extracted file already exists
            extracted_file = dir_path / base_name
            if extracted_file.exists():
                print(f"{extracted_file.name} already exists. Skipping extraction.")
                continue

            # Reconstruct the .tar.gz archive
            print(f"Reconstructing {tar_path.name} from parts...")
            with open(tar_path, "wb") as out_file:
                for part in sorted(parts):
                    print(f"Appending {part.name}")
                    with open(part, "rb") as part_file:
                        shutil.copyfileobj(part_file, out_file)

            # Extract the .tar.gz archive
            print(f"Extracting archive: {tar_path.name}")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=dir_path)
                print("Extraction complete.")

            # Remove the reconstructed archive
            tar_path.unlink()

class DictaBERTModel(NikudModel):
    def __init__(self, model_path: str, device: str, config_path: str, compressed_path: str):
        super().__init__(model_path, device, config_path, compressed_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.model.eval()

    def predict(self, sentence: str) -> str:
        return self.model.predict([sentence], self.tokenizer)[0]


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

class DNikudNikudModel(NikudModel):
    def __init__(self, model_path: str, device: str, config_path: str, compressed_path: str):
        super().__init__(model_path, device, config_path, compressed_path)

        self.logger = get_logger()

        # Load model and tokenizer
        self.tokenizer_tavbert = AutoTokenizer.from_pretrained("tau/tavbert-he")
        config = ModelConfig.load_from_file(config_path)

        self.dnikud_model = DNikudModel(config, len(Nikud.label_2_id["nikud"]), len(Nikud.label_2_id["dagesh"]),
                                   len(Nikud.label_2_id["sin"]), device=device).to(device)
        state_dict_model = self.dnikud_model.state_dict()
        state_dict_model.update(torch.load(model_path, map_location=device))
        self.dnikud_model.load_state_dict(state_dict_model)
        self.dnikud_model.eval()

    def predict(self, sentence: str) -> str:
        dataset = NikudDataset(self.tokenizer_tavbert, data_list=[sentence], logger=self.logger, max_length=MAX_LENGTH_SEN)
        dataset.prepare_data(name="prediction")
        mtb_prediction_dl = torch.utils.data.DataLoader(dataset.prepered_data, batch_size=BATCH_SIZE)
        all_labels = predict(self.dnikud_model, mtb_prediction_dl, self.device)
        return dataset.back_2_text(labels=all_labels)
