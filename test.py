import os
from datetime import datetime
import json
import pickle


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def save_history(history, abbreviation):
    """
    Save the training history to a log file in JSON format.
    """
    log_dir = f"./logs/{abbreviation}/"
    os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    history_path = os.path.join(log_dir, f"history-{timestamp}.json")

    # Convert the history to a serializable format (e.g., lists for any non-serializable objects)
    history_serializable = {
        k: [float(value) if value is not None else None for value in v]
        for k, v in history.items()
    }

    # Save the history as a JSON file
    with open(history_path, "w") as history_file:
        json.dump(history_serializable, history_file, indent=4)
    print(f"Training history saved to '{history_path}'")


svm = load_pickle("./results/svm/training log/history.pkl")

save_history(svm, "svm")

rfc = load_pickle("./results/rfc/training log/history.pkl")

save_history(rfc, "rfc")
