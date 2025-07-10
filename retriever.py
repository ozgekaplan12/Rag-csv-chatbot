from langchain.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.embeddings import SentenceTransformerEmbeddings
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL


def load_retriever(docs):
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    vector_db.add_documents(docs)
    vector_db.persist()

    sparse_retriever = BM25Retriever.from_documents(docs)
    sparse_retriever.k = 5
    dense_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )
    return hybrid_retriever