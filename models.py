
"""
Data models for the Interview Assistant API.
These define the structure of our request/response data.
"""

from pydantic import BaseModel
from typing import List

class SummaryResponse(BaseModel):
    summary: str

class QuestionsResponse(BaseModel):
    questions: List[str]
