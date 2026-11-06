# Запуск LLM-екстракції фактів з усіх документів корпусу.
import json, random, sys
from common import load_corpus, llm_structured
from schema import ExtractionResult

if hasattr(sys.stdout, "reconfigure"):
    # Windows може використовувати несумісне кодування консолі.
    sys.stdout.reconfigure(encoding="utf-8")

# Інструкції моделі та правила напрямку зв'язків у графі.
PROMPT = """Extract facts from the text for a knowledge graph.
Rules:
- Only facts explicitly stated in the text, no outside knowledge.
- Use full names ('HP Googlebook 14', not 'it' or 'the laptop').
- Directions: Product MADE_BY Company; Product POWERED_BY Chip; Product RUNS Software;
  Product/Software HAS_FEATURE Feature; Product/Software REPLACES Product/Software;
  Product COMPETES_WITH Product; Product USES_HARDWARE Hardware.

TEXT:
{text}
"""

facts, seen = [], set()
for doc_id, text in load_corpus().items():
    print("Екстракція:", doc_id)
    # Gemini повертає список Fact, перевірений Pydantic-схемою.
    for f in llm_structured(PROMPT.format(text=text), ExtractionResult).facts:
        # Нормалізований ключ використовується для видалення повторів.
        key = (f.subject.lower(), f.relation, f.object.lower())
        if key in seen:
            continue
        seen.add(key)
        # Зберігаємо також номер документа для подальшої ручної перевірки.
        facts.append({**f.model_dump(), "doc": doc_id})

# Основний результат екстракції для використання у GraphRAG.
json.dump(facts, open("facts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("Фактів:", len(facts))

# Фіксований seed робить вибірку з 20 фактів відтворюваною.
random.seed(42)
for f in random.sample(facts, min(20, len(facts))):
    print(f["doc"], "|", f["subject"], f["relation"], f["object"])