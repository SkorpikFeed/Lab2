# Спільні функції для LLM-екстракції, embedding-пошуку та чанкінгу.
import os, re, glob, time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

# API-ключ завантажується з .env, а не зберігається у вихідному коді.
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Моделі для генерації тексту та побудови embedding-векторів.
LLM_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

# Gemini обмежує кількість текстів в одному embedding-запиті.
EMBED_BATCH_SIZE = 100
# Пауза допомагає не перевищити ліміт безкоштовного рівня API.
EMBED_BATCH_INTERVAL_SECONDS = 65


def load_corpus(path="corpus"):
    """Завантажити всі TXT-документи як словник {ідентифікатор: текст}."""
    return {Path(f).stem: Path(f).read_text(encoding="utf-8")
            for f in sorted(glob.glob(os.path.join(path, "*.txt")))}


def llm_structured(prompt, schema):
    """Відповідь моделі строго за pydantic-схемою"""
    # JSON MIME type і response_schema вмикають structured output Gemini.
    r = client.models.generate_content(
        model=LLM_MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema, temperature=0))
    return r.parsed


def llm_text(prompt):
    """Звичайна текстова відповідь моделі"""
    # Використовується для резюме спільнот і фінальних RAG-відповідей.
    return client.models.generate_content(model=LLM_MODEL, contents=prompt).text


def embed(texts, kind="document"):
    """Тексти -> нормалізовані вектори з пакетною обробкою до 100 текстів."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    # Для документів і запитів Gemini використовує різні типи retrieval.
    task = "RETRIEVAL_DOCUMENT" if kind == "document" else "RETRIEVAL_QUERY"
    vectors = []
    for batch_number, start in enumerate(
        range(0, len(texts), EMBED_BATCH_SIZE)
    ):
        if batch_number:
            time.sleep(EMBED_BATCH_INTERVAL_SECONDS)
        batch = texts[start:start + EMBED_BATCH_SIZE]
        response = client.models.embed_content(
            model=EMBED_MODEL, contents=batch,
            config=types.EmbedContentConfig(task_type=task))
        # Додаємо embedding кожного тексту до спільного списку векторів.
        vectors.extend(e.values for e in response.embeddings)

    arr = np.array(vectors, dtype=np.float32)
    # Нормалізація робить скалярний добуток еквівалентом cosine similarity.
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def chunk_document(doc_id, text, max_chars=700):
    """Один абзац = один чанк; задовгий абзац ріжемо по реченню"""
    chunks = []
    # Спочатку зберігаємо межі абзаців, щоб не змішувати різні теми.
    for p in re.split(r"\n\s*\n", text.strip()):
        p = p.strip()
        while len(p) > max_chars:
            # Ріжемо довгий абзац біля останньої крапки в межах ліміту.
            cut = p.rfind(". ", 0, max_chars) + 1 or max_chars
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            chunks.append(p)
    # Метадані дозволяють повернути джерельний документ разом із результатом.
    return [{"chunk_id": f"{doc_id}#{i}", "doc_id": doc_id, "text": c}
            for i, c in enumerate(chunks)]