# Pydantic-схема фактів, які LLM видобуває з текстового корпусу.
from typing import Literal
from pydantic import BaseModel

# Дозволені типи вузлів графа знань.
EntityType = Literal["Product", "Company", "Hardware", "Feature", "Software", "Chip"]
# Дозволені типи зв'язків між вузлами.
RelationType = Literal["MADE_BY", "POWERED_BY", "RUNS", "HAS_FEATURE", "REPLACES", "COMPETES_WITH", "USES_HARDWARE"]

class Fact(BaseModel):
    subject: str
    subject_type: EntityType
    relation: RelationType
    object: str
    object_type: EntityType

class ExtractionResult(BaseModel):
    facts: list[Fact]