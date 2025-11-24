# Stock Grid View

This application provides an interactive grid view for comparing stocks. It allows users to add stocks and custom column descriptions, utilizing Google's Gemini models to generate insightful data for each cell.

## Features

-   **Interactive Grid**: Dynamic table to view and compare stock data.
-   **AI-Powered Insights**: Uses Google Gemini to generate content for custom column descriptions based on stock tickers.
-   **Caching**: Implements Redis caching to store and retrieve previously computed results for efficiency.
-   **Export**: functionality to export the grid data to Excel.

## Prerequisites

-   **Python 3.10+**
-   **Node.js & npm**
-   **Redis**: Must be installed and running for caching to work.
-   **Google Gemini API Key**: Required for generating insights.

## Getting Started

### 1. Backend Setup

Navigate to the `backend` directory:

```bash
cd backend
```

Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up environment variables:
Create a `.env` file in the `backend` directory and add your Google API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

Start the backend server:

```bash
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup

Navigate to the `frontend` directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The application should now be accessible at `http://localhost:5173`.

### 3. Redis Setup

Ensure Redis is installed and running on the default port (6379).

On macOS with Homebrew:
```bash
brew install redis
brew services start redis
```

## Disclaimer

**This application was vibecoded using Google Antigravity.**

**User assumes all risk.** The code is provided "as is" without warranty of any kind. Use at your own discretion.
