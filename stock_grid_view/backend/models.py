from pydantic import BaseModel
from typing import List, Optional, Dict

class StockRequest(BaseModel):
    ticker: str

class ColumnRequest(BaseModel):
    description: str

class GridCell(BaseModel):
    ticker: str
    column: str
    value: str

class GridResponse(BaseModel):
    stocks: List[str]
    columns: List[str]
    data: Dict[str, Dict[str, str]] # {ticker: {column: value}}
