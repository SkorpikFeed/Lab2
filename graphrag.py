# GraphRAG: граф знань, спільноти та порівняння з vector-RAG.
import json
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from common import llm_text
from vector_search import search_vector

QUESTION = "Which companies, products and technologies are central to the Googlebook story, and how are they connected?"
# Факти були отримані на попередньому етапі extract.py.
facts = json.load(open("facts.json", encoding="utf-8"))

# 1. Побудова графа: вузли — сутності, ребра — зв'язки.
G = nx.Graph()
for f in facts:
    # Атрибути ребра зберігають тип і напрямок факту.
    G.add_edge(f["subject"], f["object"], relation=f["relation"], src=f["subject"], dst=f["object"])

# 2. Виявлення спільнот Louvain і збереження зображення графа.
part = community_louvain.best_partition(G, random_state=42)
plt.figure(figsize=(14, 10))
nx.draw(G, with_labels=True, node_color=[part[n] for n in G], cmap=plt.cm.tab10, font_size=7)
plt.savefig("knowledge_graph.png", dpi=150)

# 3. GraphRAG map-reduce: резюме спільнот -> фінальна відповідь.
groups = defaultdict(list)
for node, c in part.items():
    groups[c].append(node)
summaries = []
for c, nodes in groups.items():
    # У prompt передаються лише зв'язки всередині поточної спільноти.
    edges = [f"{d['src']} -{d['relation']}-> {d['dst']}"
             for u, v, d in G.edges(data=True) if u in nodes and v in nodes]
    summaries.append(llm_text("Summarize in 3-4 sentences this group of entities and relations:\n" + "\n".join(edges)))
graphrag_answer = llm_text(
    f"Question: {QUESTION}\n\nAnswer using only these summaries:\n"
    + "\n\n".join(summaries)
)

# 4. Vector-RAG: топ-6 чанків передаються LLM без графа.
context = "\n\n".join(c["text"] for c, _ in search_vector(QUESTION, 6))
vector_answer = llm_text(f"Question: {QUESTION}\n\nAnswer using only these fragments:\n{context}")

print("=== GraphRAG ===\n", graphrag_answer)
print("\n=== vector-RAG ===\n", vector_answer)