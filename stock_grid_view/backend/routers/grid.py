from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List, Dict
import pandas as pd
import os
from ..models import StockRequest, ColumnRequest, GridResponse, GridCell
from ..services.redis_service import get_cached_cell, set_cached_cell
from ..services.gemini_service import generate_cell_content, resolve_ticker

router = APIRouter()

# In-memory state for simplicity (persisted via Redis for cell values)
# In a real app, stocks and columns would be in a DB
stocks: List[str] = []
columns: List[str] = []

@router.post("/stocks")
async def add_stock(request: StockRequest):
    ticker = resolve_ticker(request.ticker)
    if ticker not in stocks:
        stocks.append(ticker)
    return {"message": "Stock added", "stocks": stocks}

@router.post("/columns")
async def add_column(request: ColumnRequest):
    if request.description not in columns:
        columns.append(request.description)
    return {"message": "Column added", "columns": columns}

@router.get("/grid", response_model=GridResponse)
async def get_grid():
    data: Dict[str, Dict[str, str]] = {}
    
    for ticker in stocks:
        data[ticker] = {}
        for col in columns:
            # Check cache
            cached_value = get_cached_cell(ticker, col)
            if cached_value:
                data[ticker][col] = cached_value
            else:
                # Generate on demand
                value = generate_cell_content(ticker, col)
                # Only cache if valid response
                if value not in ["Error", "Set API Key"]:
                    set_cached_cell(ticker, col, value)
                data[ticker][col] = value
                
    return GridResponse(stocks=stocks, columns=columns, data=data)

@router.post("/refresh", response_model=GridResponse)
async def refresh_grid():
    """Recompute all cells, bypassing cache."""
    data: Dict[str, Dict[str, str]] = {}
    
    for ticker in stocks:
        data[ticker] = {}
        for col in columns:
            value = generate_cell_content(ticker, col)
            if value not in ["Error", "Set API Key"]:
                set_cached_cell(ticker, col, value)
            data[ticker][col] = value
            
    return GridResponse(stocks=stocks, columns=columns, data=data)

@router.get("/export")
async def export_grid():
    # Create DataFrame
    data = []
    for ticker in stocks:
        row = {"Ticker": ticker}
        for col in columns:
            row[col] = get_cached_cell(ticker, col) or "N/A"
        data.append(row)
    
    df = pd.DataFrame(data)
    filename = "stock_grid.xlsx"
    df.to_excel(filename, index=False)
    
    return FileResponse(filename, filename=filename, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
