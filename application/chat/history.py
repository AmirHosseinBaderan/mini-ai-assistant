from typing import Literal

Role = Literal["user","assistant"]

class ConversationHistory:
    def __init__(self):
        self._messages:list[dict[str,str]] = []
        
    def add_user(self,content:str):
        self._messages.append({
            "role":"user",
            "content": content
        })
    
    def add_assistant(self,content:str):
        self._messages.append({
            "role":"assistant",
            "content":content
        })
        
    def get_messages(self)-> list[dict[str,str]]:
        return self._messages.copy()
    
    def clear(self):
        self._messages.clear()