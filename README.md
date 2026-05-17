# Stock Analyser

A comprehensive AI-powered stock analysis system built with CrewAI that generates detailed equity research reports. The system uses 20+ specialized AI agents to analyze stocks from multiple perspectives: fundamental analysis, technical analysis, competitive positioning, macroeconomic context, risk assessment, and investment recommendations styled after six famous investors.

**Version**: 0.2.0 | **Python**: 3.12.9 (3.13+ not supported)

---

## Features

- **Multi-Agent Orchestration**: 22 specialized CrewAI crews covering every aspect of equity research
- **Full-Length Reports**: Generates comprehensive markdown + PDF reports (typically 40–80 pages)
- **Technical Analysis Suite**: 10+ chart types — MACD/Stochastic, RSI, Squeeze Momentum, Supertrend, Standard Error Bands, Multi-Timeframe Momentum, Awesome Oscillator, Breakout, and more
- **Fundamental Analysis**: Deep dive into income statements, balance sheets, cash flows, DCF sensitivity, and ratio trends
- **Competitive Intelligence**: Automated competitor identification, price comparison charts, and landscape analysis
- **SEC Filings Integration**: Downloads and semantically searches official filings using Gemini embeddings + Qdrant vector DB
- **Six Investor Perspectives**: Analysis in the style of Warren Buffett, Charlie Munger, Benjamin Graham, Cathie Wood, Stanley Druckenmiller, and Bill Ackman
- **Risk Assessment**: Risk identification, quantification, and severity visualization
- **Automated Distribution**: Creates Gmail draft with PDF attachment ready to send

---

## Architecture

The system is built around a **CrewAI Flow** (`ReportFlow`) that orchestrates the full pipeline:

```
kickoff()
  │
  ├─ [prompt] Enter stock ticker  ← runs BEFORE flow starts
  │
  └─ ReportFlow.kickoff()
       │
       ├─ confirm_company_stock   ← validate ticker via YFinance
       ├─ set_up                  ← download SEC filings → Qdrant embeddings
       │
       ├─ generate_news_and_research_section
       ├─ generate_market_and_industry_section
       ├─ identify_competitors
       ├─ generate_competitor_landscape_section
       │    ├─ plot_competitor_prices          ─┐ parallel
       │    └─ generate_financial_data_section ─┘
       │         ├─ plot_historical_stock_data
       │         └─ generate_trends_section
       │              └─ generate_analyst_insights_section
       │                   ├─ plot_signals_charts
       │                   └─ plot_dcf_sensitivity_3D_chart
       │                        └─ [6 investor crews, sequential]
       │                              └─ generate_risk_analysis_section
       │                                   ├─ plot_risk_severity
       │                                   └─ generate_future_outlook_section
       │                                        └─ generate_investment_recommendations_section
       │                                             └─ generate_executive_summary
       │                                                  └─ plot_candlestick_30_days
       │                                                       └─ generate_conclusion
       │                                                            └─ save_markdown_report
       │                                                                 ├─ convert_report_to_pdf
       │                                                                 └─ (wait for both)
       │                                                                      └─ draft_email_with_attachment
       │                                                                           └─ print_total_token_usage
```

### Specialized Analysis Crews

| Crew | Analysis |
|------|---------|
| News & Research | Recent news, press releases, market research |
| Market & Industry | Sector context, industry positioning, macro trends |
| Competition Identifier | Identifies competitor ticker symbols |
| Competitor | Deep competitive landscape, SWOT, moat analysis |
| Financial Data | Financial statements, ratios, DCF valuation |
| Trends | Revenue/income/margin/EPS/FCF trend analysis |
| Analyst Insights | Sell-side ratings, price targets, estimates |
| Warren Buffett | Moat, ROE, earnings quality, margin of safety |
| Charlie Munger | Mental models, quality at fair price |
| Benjamin Graham | Net-net value, P/E, defensive criteria |
| Cathie Wood | Disruptive innovation, TAM, growth trajectory |
| Stanley Druckenmiller | Macro environment, momentum, positioning |
| Bill Ackman | Activist lens, capital allocation, FCF yield |
| Risk Analysis | Business, financial, market, regulatory risks |
| Risk Severity | Structured risk severity scoring |
| Investment Recommendation | Buy/Hold/Sell with price target synthesis |
| Future Outlook | Growth prospects, strategic initiatives, catalysts |
| Executive Summary | High-level synthesis of full report |
| Conclusion | Final takeaways + legal disclaimer |
| Chart Reading | Interprets chart images |
| Gmail Attachment | Creates Gmail draft with PDF |

---

## Prerequisites

- Python 3.12.9 (3.12.x required — 3.13+ not supported due to dependency constraints)
- A virtual environment
- API keys (see Environment Setup)
- Google OAuth2 `credentials.json` for Gmail integration (optional)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jasperb3/StockAnalyser.git
cd StockAnalyser
```

2. Create and activate a virtual environment:
```bash
python3.12 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
```

3. Install dependencies:
```bash
pip install -e .
```

4. Copy and populate the environment file:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

---

## Environment Setup

### Required

**LLM (at least one)**:
| Variable | Provider |
|----------|---------|
| `GEMINI_API_KEY` | Google Gemini (recommended — used for embeddings) |
| `OPENAI_API_KEY` | OpenAI GPT-4.1 / GPT-5 family |
| `GROQ_API_KEY` | Groq (Llama 4 Maverick / Scout) |
| `MISTRAL_API_KEY` | Mistral Large |

**Vector Database** (pick local or cloud):
| Variable | Description |
|----------|-------------|
| `QDRANT_LOCAL_URL` | Local Qdrant instance URL |
| `QDRANT_LOCAL_API_KEY` | Local Qdrant API key |
| `QDRANT_CLUSTER_URL` | Qdrant cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant cloud API key |
| `QDRANT_COLLECTION_NAME` | Collection name for SEC filing embeddings |

**Data APIs**:
| Variable | Provider |
|----------|---------|
| `FMP_API_KEY` | Financial Modeling Prep (SEC filings download) |
| `SEC_API_KEY` | SEC API direct access |
| `TAVILY_API_KEY` | Tavily AI search |
| `LINKUP_API_KEY` | Linkup search |
| `EXA_API_KEY` | Exa search |
| `GOOGLE_SEARCH_API_KEY` + `GOOGLE_CSE_ID` | Google Custom Search |
| `SERPER_API_KEY` | Serper search |
| `NASDAQ_API_KEY` | Nasdaq Data Link |

### Optional

**Email**:
| Variable | Description |
|----------|-------------|
| `CLIENT` | Recipient name for report header |
| `SENDER` | Sender Gmail address |

Gmail also requires `credentials.json` (OAuth2 client secret) placed in `src/stock_analyser/tools/`. See [Gmail OAuth Setup](#gmail-oauth-setup).

---

## Usage

### Run Full Analysis

```bash
uv run kickoff
# or
.venv/bin/python -m stock_analyser.main
```

You will be prompted to enter a stock ticker before the flow starts:
```
Enter the stock ticker symbol of the company you want to analyze or hit 'ENTER' for a random S&P 500 company:
```

Enter a ticker (e.g. `AAPL`, `NVDA`) or press Enter to pick a random S&P 500 stock.

### Generate Plots Only

```bash
uv run plot
# or
.venv/bin/python -m stock_analyser.main plot
```

---

## Output

| Type | Location | Format |
|------|----------|--------|
| Full report | `final_reports/<date>/` | `.md` + `.pdf` |
| Technical charts | `final_reports/<date>/plots/` | `.png` |
| Timing/token log | `timings/timeit_<TIMESTAMP>.txt` | `.txt` |
| Crew raw outputs | `crewOutputs/<TIMESTAMP>/` | `.md` |
| SEC filings | `filings/` | `.md` |
| Gmail draft | Gmail Drafts folder | ready-to-send |

---

## Technical Analysis Charts

The system generates and embeds the following charts in the report:

| Chart | Indicators |
|-------|-----------|
| MACD + Stochastic | Candlesticks, volume, MACD histogram, signal line, Stochastic %K/%D |
| Relative Strength + RSI | Closing price, MAs (10/20/50/100), volume, RS vs S&P 500, RSI (14) |
| Squeeze Momentum | Bollinger Bands, Keltner Channels, momentum histogram, squeeze dots |
| Supertrend | ATR-based trend lines, buy/sell signals, backtest ROI |
| Standard Error Bands | Linear regression ± 2 standard errors channel |
| Multi-Timeframe Momentum | RSI, Williams %R, Ultimate Oscillator, Schaff Trend across D/W/M |
| DCF Sensitivity (3D) | Implied price surface across WACC and Terminal Growth Rate scenarios |
| Competitor Prices | Normalized price comparison across all identified competitors |
| Risk Severity | Bubble chart of identified risks by probability × impact |
| Candlestick (30-day) | OHLC candlesticks with volume for the most recent month |

---

## Project Structure

```
stock_analyser/
├── src/stock_analyser/
│   ├── main.py                    # ReportFlow orchestration + kickoff()
│   ├── crews/                     # 22 specialized analysis crews
│   │   └── <name>_crew/
│   │       ├── <name>_crew.py
│   │       └── config/
│   │           ├── agents.yaml
│   │           └── tasks.yaml
│   ├── tools/                     # 40+ data gathering tools
│   │   ├── yfinance_*.py          # Yahoo Finance wrappers
│   │   ├── gemini_*_search_tool.py
│   │   ├── gmail_utility_with_attachment.py  # OAuth2 Gmail
│   │   ├── qdrant_sec_filings_search_tool.py
│   │   ├── credentials.json       # Gmail OAuth client secret
│   │   ├── token.json             # Gmail OAuth token (auto-managed)
│   │   └── tool_utils/            # Investor analysis implementations
│   │       ├── warren_buffett_analysis.py
│   │       ├── ben_graham_analysis.py
│   │       ├── intrinsic_value_dcf.py
│   │       └── ...
│   ├── plotting/                  # 20+ chart generation functions
│   │   ├── plot_macd_stochastic.py
│   │   ├── plot_squeeze_momentum.py
│   │   ├── plot_multi_timeframe_momentum.py
│   │   └── ...
│   └── utils/
│       ├── llms.py                # LLM instance definitions
│       ├── embeddings_fn.py       # Gemini embedding function
│       ├── setup.py               # Setup class (SEC → Qdrant pipeline)
│       ├── models.py              # ReportState Pydantic model
│       ├── constants.py           # Paths, timestamps, BACKOFF_TIME
│       ├── get_sp500_tickers.py   # Wikipedia S&P 500 list (requests)
│       ├── convert_to_pdf.py      # WeasyPrint markdown → PDF
│       ├── screener.py            # Stock screening utilities
│       └── styling.css            # PDF report CSS
├── tests/
│   ├── test_qdrant_tool.py
│   └── test_investor_analyses.py
├── final_reports/                 # Generated reports (gitignored)
├── filings/                       # Downloaded SEC filings (gitignored)
├── timings/                       # Performance logs (gitignored)
├── .env.example                   # Environment template
├── pyproject.toml                 # Dependencies + entry points
└── README.md
```

---

## Gmail OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials
2. Create an OAuth 2.0 Client ID (Desktop application)
3. Download `credentials.json` and place it in `src/stock_analyser/tools/`
4. On first run, a browser window opens for authorization — `token.json` is saved automatically
5. If `token.json` expires or is revoked, it is deleted automatically and re-authorization is triggered

> **Important**: If your OAuth app is in "Testing" mode, refresh tokens expire after 7 days. Either publish the app to production status or set up a service account to avoid repeated re-authorization.

---

## Development

### Running Tests

```bash
pytest
pytest tests/test_investor_analyses.py  # specific file
```

### Adding New Analysis Crews

1. Create `src/stock_analyser/crews/<name>_crew/`
2. Add `config/agents.yaml` and `config/tasks.yaml`
3. Create crew class with `@crew()` method
4. Add a `@listen()` decorated step in `main.py`

### Adding New Tools

1. Create `src/stock_analyser/tools/<name>_tool.py`
2. Inherit from `crewai.tools.BaseTool`
3. Implement `_run()` method
4. Add to appropriate crew's `tasks.yaml` under `tools:`

### Adding New Plots

1. Create `src/stock_analyser/plotting/plot_<name>.py`
2. Return the saved file path from the main function
3. Call from the appropriate flow step in `main.py`
4. Inject the path via `.replace("[placeholder]", ...)` into the relevant section string

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ImportError` | Not installed in dev mode | `pip install -e .` |
| `HTTP 403` fetching S&P 500 list | Wikipedia blocking urllib | Already fixed in `get_sp500_tickers.py` |
| Input prompt buried in terminal | CrewAI UI overlays | Already fixed — prompt runs before flow starts |
| `invalid_grant: Bad Request` (Gmail) | Expired/revoked refresh token | Already fixed — auto re-auth; browser will open |
| `QDRANT_LOCAL_URL not found` | Wrong env var name | Use `QDRANT_LOCAL_URL` (not `QDRANT_CLUSTER_URL`) for local |
| Gemini rate limit errors | Too many API calls in succession | `BACKOFF_TIME` pauses are built in; increase if needed |
| Memory issues on large filings | Large SEC document batches | Adjust batch sizes in `utils/setup.py` |
| `ta-lib` install fails | Requires C library | Install `libta-lib-dev` first: `sudo apt install libta-lib-dev` |
| PDF generation fails | WeasyPrint / CSS issue | Check `utils/styling.css` and `utils/convert_to_pdf.py` |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-analysis`
3. Commit using conventional commits: `git commit -m "feat: add X analysis crew"`
4. Push and open a Pull Request

---

## Key Technologies

| Technology | Role |
|-----------|------|
| [CrewAI](https://github.com/joaomdmoura/crewAI) | Multi-agent orchestration and flow management |
| [Google Gemini](https://ai.google.dev/) | Primary LLM + embeddings (`gemini-embedding-001`) |
| [Qdrant](https://qdrant.tech/) | Vector database for SEC filing semantic search |
| [YFinance](https://github.com/ranaroussi/yfinance) | Financial data (prices, financials, ratios) |
| [Matplotlib / mplfinance](https://github.com/matplotlib/mplfinance) | Technical and financial chart generation |
| [Plotly](https://plotly.com/) | 3D DCF sensitivity surface chart |
| [Pydantic](https://docs.pydantic.dev/) | Type-safe `ReportState` model |
| [WeasyPrint](https://weasyprint.org/) | HTML/CSS → PDF report rendering |
| [sec-api / FMP](https://sec-api.io/) | SEC filing downloads |
| [pandas-ta / TA-Lib / ta](https://github.com/twopirllc/pandas-ta) | Technical indicator calculations |

---

## License

This project is private. All rights reserved.
