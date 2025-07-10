import glob
import pandas as pd
from langchain.schema import Document
from retriever import load_retriever
from llm_infer import get_llm
from reranker import rerank

# CSV'den metinleri oku
df = pd.read_csv("./df_file.csv")  # veya ./data/df_file.csv, nereye koyduğuna göre

# Her satırı bir Document olarak hazırla
manual_docs = [
    Document(
        page_content=row["Text"], 
        metadata={"label": str(row["Label"])}
    )
    for _, row in df.iterrows()
]

retriever = load_retriever(manual_docs)
llm = get_llm()

query = "What measures are being proposed regarding stamp duty in the budget?"
docs = retriever.invoke(query)
reranked = rerank(query, docs)

context = "\n\n".join([
    f"{doc.page_content}\n\nKategori: {doc.metadata.get('label', '')}" for doc in reranked[:3]
])


prompt = f"""
You are answering the user's question based only on the information provided below. Be clear, concise, and to the point.

Rules:
- Only use information that directly answers the question.
- Do not include unnecessary details.
- Keep your answer clear, simple, and focused.
- Stay faithful to the content, but avoid overexplaining.
- Do not guess or add external knowledge.
- Answer only what is asked. Do not include additional information.
- Do not use technical terms like "context", "source", or "retrieved".
- If there is no relevant information, respond with: "No information available on this topic."
- If the question is unclear, ask the user to clarify.

Question:
{query}

Information:
{context}

Answer:
"""

response = llm.invoke(prompt)
print(response)
