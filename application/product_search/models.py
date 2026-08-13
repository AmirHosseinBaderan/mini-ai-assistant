from dataclasses import dataclass

@dataclass(frozen=True)
class SiteConfig:
    name:str
    search_url:str