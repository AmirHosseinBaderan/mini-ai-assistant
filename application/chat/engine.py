from application.llm.client import LLMClient

class ChatEngine:
    def __init__(self,llm:LLMClient):
        self.llm = llm
        self.messages:list[dict[str,str]] = []
        
    def chat(
        self,
        user_message:str
    )-> str:
        self.messages.append({
            "role":"user",
            "content":user_message
        })
        
        response = []
        for chunk in self.llm.stream(
            self.messages
        ):
            print(chunk,end="",flush=True)
            response.append(chunk)
        
        print()
        assistant_message = "".join(response)
        self.messages.append({
            "role":"assistant",
            "content":assistant_message
        })
        
        return assistant_message
        