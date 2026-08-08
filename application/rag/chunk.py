from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)