
import yfinance as yf
from data_model import StockReport, BulletPoint
from typing import List

async def get_stock_price(ticker) -> float:
    """Gets the latest price of a stock using yfinance."""
    print(f"Calling Yahoo Finance for price of {ticker}")
    stock = yf.Ticker(ticker)
    todays_data = stock.history(period='1d')
    print(todays_data)
    return todays_data['Close'][0]

async def get_stock_price_mock(ticker) -> float:
    return 236.40


def render_report(report: StockReport) -> str:

    def format_bullets(bullets: List[BulletPoint]) -> str:
        result = ""
        for bullet in bullets:
            result += f"* **{bullet.title}** {bullet.explanation}\n"
        return result

    newl = "\n   "
    result = f"""
## Analysis report for {report.company.name} ({report.company.ticker}) at {report.company.latest_price}

### Company Overview
{report.company_overview.what_company_does}
**Industry:** {report.company_overview.industry}
**Products:** 
    {newl.join(report.company_overview.key_products_or_segments)}

### Key Financial Data
**Market cap:** {report.key_financial_data.market_cap}
**Total revenue:** {report.key_financial_data.revenue}
**Net income:** {report.key_financial_data.net_income}
**PE ratio:** {report.key_financial_data.pe_ratio}
**Dividend yield:** {report.key_financial_data.dividend_yield}

### Buy Case
{format_bullets(report.buy_case)}

### Sell Case
{format_bullets(report.sell_case)}

    """
    return result

