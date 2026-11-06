# Побудова індексу та порівняння TF-IDF із семантичним пошуком.
import sys, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import load_corpus, chunk_document, embed

if hasattr(sys.stdout, "reconfigure"):
    # Дозволяє виводити українські повідомлення у Windows-консолі.
    sys.stdout.reconfigure(encoding="utf-8")


def build_index():
    """Створити та зберегти текстовий і embedding-індекси корпусу."""
    chunks = []
    for doc_id, text in load_corpus().items():
        # Кожен документ перетворюється на набір незалежних чанків.
        chunks += chunk_document(doc_id, text)
    # Зберігаємо тексти й метадані окремо від числових embedding-векторів.
    json.dump(chunks, open("chunks.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    np.save("embeddings.npy", embed([c["text"] for c in chunks]))
    print("Чанків:", len(chunks))


def _load():
    """Завантажити індекс без повторного виклику embedding API."""
    return json.load(open("chunks.json", encoding="utf-8")), np.load("embeddings.npy")


def search_vector(query, n=3):
    """запит -> топ-N чанків за косинусною схожістю"""
    chunks, emb = _load()
    sims = emb @ embed([query], "query")[0]
    return [(chunks[i], float(sims[i])) for i in np.argsort(-sims)[:n]]


def search_keyword(query, n=3):
    """TF-IDF: пошук за збігом слів"""
    chunks, _ = _load()
    # TF-IDF є baseline для порівняння з семантичним пошуком.
    vec = TfidfVectorizer()
    X = vec.fit_transform([c["text"] for c in chunks])
    sims = (X @ vec.transform([query]).T).toarray().ravel()
    return [(chunks[i], float(sims[i])) for i in np.argsort(-sims)[:n]]


QUERIES = [
    # Simple questions: the answer is stated directly in one document.
    "According to the Lenovo Googlebook 15 specification, which processor is listed?",
    "Which feature allows the Acer Googlebook 14 to fold into different positions?",
    "What starting price is stated for Googlebook pre-orders?",
    # Synthesis questions: the answer requires comparing multiple documents.
    "Which companies manufacture Googlebook devices, and which processors do they use?",
    "Which Googlebook features repeatedly connect the laptop with Android phones and services?",
    "How is Googlebook positioned against Chromebooks, MacBook Air, and Surface Laptop?",
]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    else:
        # Для кожного питання виводимо результати обох методів.
        for q in QUERIES:
            print("\n=== ", q)
            for name, fn in (("TF-IDF", search_keyword), ("Semantic", search_vector)):
                print(name)
                for c, s in fn(q):
                    print(f"  {s:.3f} {c['chunk_id']}: {c['text'][:100]}")