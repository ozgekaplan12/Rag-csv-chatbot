# 🧠 LLM Destekli Soru-Cevap Sistemi (Kullanım Kılavuzu Üzerinden)

Bu proje, csv formatında etiketlenmiş kullanım kılavuzu verilerini kullanarak, bir soruya LLM (Large Language Model) desteğiyle en alakalı ve kısa cevabı vermeyi amaçlar.


## 🚀 Özellikler

- `RAG (Retrieval-Augmented Generation)` mimarisi kullanır.
- `BM25 + Dense` hybrid arama ile en ilgili dökümanları bulur.
- `Cross-Encoder` ile yeniden sıralama (reranking) yapar.
- LLM yanıtı, yalnızca içeriğe dayanarak üretir. Tahmin içermez.

## 📁 Proje Yapısı

.
├── df_file.csv # Etiketli veriler
├── config.py # Model ve yol ayarları
├── retriever.py # Arama (retrieval) işlemleri
├── reranker.py # Reranking işlemi
├── llm_infer.py # LLM ile yanıt üretimi
├── main.py # Ana çalışma dosyası
├── requirements.txt # Gerekli kütüphaneler
└── README.md # Bu dosya :)


## 📦 Gereksinimler

```bash
pip install -r requirements.txt

Not: Proje LangChain, ChromaDB, SentenceTransformers, Ollama, Transformers gibi kütüphaneleri kullanır.
Ollama kurulu değilse: https://ollama.com/download

# sanal ortam aktifken:
python main.py


