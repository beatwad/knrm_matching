from typing import Dict, List

import os
import json
import torch
from flask import Flask, request, jsonify
from langdetect import detect
from knrm import KNRM
import string
import nltk
import threading
import dotenv


dotenv.load_dotenv()

app = Flask(__name__)

helper = None
faiss_is_ready = False


class Helper:
    def __init__(self):
        self.emb_path_glove = os.getenv("EMB_PATH_GLOVE")
        self.vocab_path = os.getenv("VOCAB_PATH")
        self.emb_path_knrm = os.getenv("EMB_PATH_KNRM")
        self.mlp_path = os.getenv("MLP_PATH")

        self.prepare_model()
        self.glove_embeddings = self._read_glove_embeddings(self.emb_path_glove)
        self.vocab = self._load_vocab(self.vocab_path)

    def prepare_model(self):
        self.model = KNRM(
            emb_state_dict=torch.load(self.emb_path_knrm),
            freeze_embeddings=True,
            out_layers=[],
            kernel_num=21,
            sigma=0.1,
            exact_sigma=0.001,
        )
        self.model.mlp.load_state_dict(torch.load(self.mlp_path))

        global model_is_ready
        model_is_ready = True

    def handle_punctuation(self, inp_str: str) -> str:
        for symbol in string.punctuation:
            inp_str = inp_str.replace(symbol, " ")
        return inp_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        inp_str = self.handle_punctuation(inp_str)
        inp_str = inp_str.lower()
        return nltk.word_tokenize(inp_str)

    def prepoc_query(self, query: List[str]):
        query = list(map(self.simple_preproc, query))
        return query

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        glove_embeddings = {}
        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                vector = [float(x) for x in parts[1:]]
                glove_embeddings[word] = vector
        return glove_embeddings

    def _load_vocab(self, file_path: str) -> Dict[str, int]:
        with open(file_path, "r") as f_in:
            vocab = json.load(f_in)
            return vocab


def init_helper():
    global helper
    helper = Helper()


with app.app_context():
    thread = threading.Thread(target=init_helper)
    thread.daemon = True
    thread.start()


@app.route("/ping")
def ping():
    if helper:
        return jsonify(status="ok")
    return jsonify(status="in_progress")


@app.route("/query", methods=["POST"])
def query():
    # TODO: return json with status='FAISS is not initialized!' if FAISS index was not loaded
    data = request.get_json()
    queries = data["queries"]
    lang_check = []
    suggestions = []
    for q in queries:
        is_en = detect(q) == "en"
        lang_check.append(is_en)
        if not is_en:
            suggestions.append(None)
            continue
        processed_query = helper.simple_preproc(q)


@app.route("/update_index", methods=["POST"])
def update_index():
    data = request.get_json()
    documents = data["documents"]
    # TODO: generate FAISS index here


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=11000)
