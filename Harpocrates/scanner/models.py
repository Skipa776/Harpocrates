from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class Finding:
    '''
    A single Potential secret found in a file
    '''
    scanner_name: str
    file_path: str
    line_number: int
    column: int
    raw_text: str
    masked_text: str
    signature_name: str
    confidence_score : float = field(default=0.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def enhance_confidence(self, amount: float) -> None:
        '''
        Adjust confidence by `amount`, clamped into [0.0, 1.0].

        Positive values increase confidence, negative decrease it.
        '''
        self.confidence_score = max(0.0, min(1.0, self.confidence_score + amount))
        