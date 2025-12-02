from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .models import Finding

class BaseScanner(ABC):
    '''
    Abstract base class for all secret scanners.
    
    Subclasses must implement the `scan` method to analyze content
    and return a list of Finding instances.
    '''
    
    def __init__(self, name: str | None = None, **kwargs: Any)-> None:
        self.name = name or self.__class__.__name__
        self._init_kwargs = kwargs
    
    @abstractmethod
    def scan(self, content: str, context: Dict[str, Any]) -> List[Finding]:
        """
        Scan the given `content` and return a list of findings.

        Parameters
        ----------
        content:
            The full file content as a single string.
        context:
            Extra information about the scan target. Typical keys:
              - "file_path": path to the file being scanned (str or Path)
              - anything else you may need (config, thresholds, etc.)

        Returns
        -------
        List[Finding]:
            Zero or more Finding instances.
        """
        raise NotImplementedError()