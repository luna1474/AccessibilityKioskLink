#for Msg
from dataclasses import dataclass

@dataclass
class Msg:
    msg:str = None
    photo = None
    idx: int = None
    x: int = None
    y: int = None