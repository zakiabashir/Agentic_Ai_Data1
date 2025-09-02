from pydantic import BaseModel
class OutputCheck(BaseModel):
    is_political: bool
    reason: str
class MathOutPut(BaseModel):
    is_math: bool
    reason: str