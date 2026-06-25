from sentence_transformers import SentenceTransformer
import os

MOC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_MODEL_PATH = os.path.join(MOC_ROOT, "models", "all-MiniLM-L6-v2")

_model = None

def get_sentence_embedding(sentence):
    global _model
    if _model is None:
        _model = SentenceTransformer(LOCAL_MODEL_PATH, local_files_only=True)
    embeddings = _model.encode(sentence)
    return embeddings
