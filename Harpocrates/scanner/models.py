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
        """
        Adjust the finding's confidence score by a given amount and clamp the result to the range 0.0–1.0.
        
        Parameters:
            amount (float): Amount to add to the current confidence score; positive values increase confidence, negative values decrease it. The confidence_score is updated in place.
        """
        new_value = self.confidence_score + amount
        if new_value < 0.0:
            new_value = 0.0
        elif new_value > 1.0:
            new_value = 1.0
        self.confidence_score = new_value
        