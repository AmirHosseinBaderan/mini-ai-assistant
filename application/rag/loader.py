from pathlib import Path
from application.rag.document import Document

class DocumentLoader:
    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md"
    }
    
    def load(self,path:str|Path) -> Document:
        path = Path(path)
        if not path.exists()