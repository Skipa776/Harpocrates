from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .models import Finding

class BaseScanner(ABC):
    '''
    Docstring for BaseScanner
    '''
    
    def __init__(self, name: str | None = None, **kwargs: Any)-> None:
        """
        Initialize the scanner instance.
        
        Parameters:
            name (str | None): Optional name for the scanner; if omitted, the class name is used.
            **kwargs: Additional initialization keyword arguments stored on the instance as `__init__kwargs`.
        """
        self.name = name or self.__class__.__name__
        self.__init__kwargs = kwargs
    
    @abstractmethod
    def scan(self, content: str, context: Dict[str, Any]) -> List[Finding]:
        """
        Scan provided content and identify findings.
        
        Parameters:
            content (str): The full file content as a single string.
            context (Dict[str, Any]): Additional information about the scan target. Common keys include:
                - "file_path": path to the file being scanned (str or Path)
                - configuration, thresholds, or other scan-specific options
        
        Returns:
            List[Finding]: Zero or more Finding instances detected in the content.
        """
        raise NotImplementedError()