from typing import List
from pydantic import BaseModel, Field


class Company(BaseModel):
    name: str
    ticker: str
    latest_price: float = Field(description="The latest price of the stock.")


class CompanyOverview(BaseModel):
    industry: str
    what_company_does: str = Field(description="Description of the company's business.")
    key_products_or_segments: List[str] = Field(description="List of key products or segments.")


class KeyFinancialData(BaseModel):
    market_cap: str = Field(description="Market capitalization")
    revenue: str = Field(description="Total revenue")
    net_income: str = Field(description="Net income")
    pe_ratio: str = Field(description="Price-to-earnings ratio")
    dividend_yield: str = Field(description="Dividend yield")


class BulletPoint(BaseModel):
    title: str
    explanation: str


class StockReport(BaseModel):
    company: Company
    company_overview: CompanyOverview
    key_financial_data: KeyFinancialData
    buy_case: List[BulletPoint]
    sell_case: List[BulletPoint]


