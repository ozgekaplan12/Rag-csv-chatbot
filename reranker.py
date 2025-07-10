from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import RERANKER_MODEL


tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)

def rerank(query, docs):
    pairs = [(query, doc.page_content) for doc in docs]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: -x[0])]
    return sorted_docs