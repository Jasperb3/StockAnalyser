# Stock Analyser

A comprehensive AI-powered stock analysis system built with CrewAI that generates detailed equity research reports. The system uses multiple specialized AI agents to analyze stocks from various perspectives including fundamental analysis, technical analysis, competitive positioning, and investment recommendations styled after famous investors.

## Features

- **Multi-Agent Analysis**: Specialized AI crews for different aspects of stock analysis
- **Comprehensive Reports**: Generates detailed markdown and PDF reports
- **Technical Analysis**: Advanced charting with multiple indicators (MACD, RSI, Squeeze Momentum, Supertrend)
- **Fundamental Analysis**: Deep dive into financial statements, ratios, and trends
- **Competitive Intelligence**: Automated competitor identification and analysis
- **SEC Filings Integration**: Downloads and searches official company filings using vector embeddings
- **Investment Styles**: Analysis from perspectives of Warren Buffett, Charlie Munger, Benjamin Graham, Cathie Wood, Stanley Druckenmiller, and Bill Ackman
- **Risk Assessment**: Comprehensive risk analysis and severity scoring
- **Automated Distribution**: Email integration for report delivery

## Architecture

The system is built around a **CrewAI Flow** that orchestrates the analysis pipeline:

1. **Input**: User provides stock ticker or selects random S&P 500 stock
2. **Setup**: Downloads SEC filings and stores in Qdrant vector database
3. **Analysis**: Multiple specialized crews analyze different aspects sequentially
4. **Visualization**: Generates technical and financial charts
5. **Report Generation**: Combines all sections into comprehensive reports
6. **Distribution**: Creates draft email with report attachment

### Specialized Analysis Crews

- **News & Research**: Recent news and market research
- **Market & Industry**: Market context and industry positioning
- **Competition**: Competitor identification and competitive landscape
- **Financial Data**: Financial statements and metrics analysis
- **Trends**: Financial metric trends over time
- **Analyst Insights**: Analyst recommendations and valuations
- **Investment Styles**: Multi-perspective investment analysis
- **Risk Analysis**: Investment risk identification and quantification
- **Future Outlook**: Growth prospects and strategic initiatives
- **Executive Summary**: Synthesis of all findings

## Prerequisites

- Python 3.12.9 (3.12.x required, 3.13+ not supported)
- Virtual environment (recommended)
- Multiple API keys (see Environment Setup)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jasperb3/StockAnalyser.git
cd StockAnalyser
```

2. Create and activate a virtual environment:
```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:

### Required API Keys

**LLM Providers** (at least one required):
- `GEMINI_API_KEY` - Google Gemini (recommended for embeddings)
- `OPENAI_API_KEY` - OpenAI GPT models
- `ANTHROPIC_API_KEY` - Anthropic Claude
- `GROQ_API_KEY` - Groq
- `MISTRAL_API_KEY` - Mistral AI
- `DEEPSEEK_API_KEY` - DeepSeek
- `PERPLEXITY_API_KEY` - Perplexity
- `OPENROUTER_API_KEY` - OpenRouter

**Data & Search APIs**:
- `SEC_API_KEY` - SEC filings access
- `FMP_API_KEY` - Financial Modeling Prep
- `TAVILY_API_KEY` - Tavily search
- `LINKUP_API_KEY` - Linkup search
- `NASDAQ_API_KEY` - Nasdaq Data Link
- `EXA_API_KEY` - Exa search
- `GOOGLE_SEARCH_API_KEY` & `GOOGLE_CSE_ID` - Google Custom Search
- `SERPER_API_KEY` - Serper search

**Vector Database**:
- `QDRANT_CLUSTER_URL` - Qdrant cluster URL
- `QDRANT_API_KEY` - Qdrant API key
- `QDRANT_COLLECTION_NAME` - Collection name for embeddings

**Email** (optional):
- `CLIENT` - Email client identifier
- `SENDER` - Sender email address

## Usage

### Run Full Analysis

```bash
python -m stock_analyser.main
# or
kickoff
```

You'll be prompted to enter a stock ticker or press Enter for a random S&P 500 stock.

### Generate Plots Only

```bash
python -m stock_analyser.main plot
# or
plot
```

### Command Line Options

The system supports interactive prompts for:
- Stock ticker selection
- Random S&P 500 stock selection
- Plot generation mode

## Output

The system generates:

- **Reports**: Markdown and PDF format in `final_reports/`
- **Charts**: Technical and financial visualizations in `archived/plots/`
- **Timing Data**: Performance metrics in `timings/`
- **Crew Outputs**: Intermediate analysis in `crewOutputs/`
- **Email Drafts**: Ready-to-send reports with attachments

## Project Structure

```
stock_analyser/
├── src/stock_analyser/
│   ├── crews/              # Specialized analysis crews
│   ├── tools/              # Data gathering and analysis tools
│   ├── plotting/           # Chart generation functions
│   ├── utils/              # Utilities (LLMs, setup, models)
│   └── main.py            # Main flow orchestration
├── tests/                  # Test files
├── .env.example           # Environment template
├── pyproject.toml         # Project dependencies
└── README.md              # This file
```

## Key Technologies

- **CrewAI**: Multi-agent orchestration framework
- **Qdrant**: Vector database for SEC filings
- **YFinance**: Financial data retrieval
- **Matplotlib/mplfinance**: Chart generation
- **Pydantic**: Type-safe state management
- **Google Gemini**: Embeddings and analysis
- **WeasyPrint**: PDF report generation

## Development

### Running Tests

```bash
pytest
```

### Adding New Analysis Crews

1. Create crew directory in `src/stock_analyser/crews/`
2. Add `config/agents.yaml` and `config/tasks.yaml`
3. Create crew class inheriting from `crewai.Crew`
4. Add to flow in `main.py` with `@listen()` decorator

### Adding New Tools

1. Create tool in `src/stock_analyser/tools/`
2. Inherit from `crewai.tools.BaseTool`
3. Implement `_run()` method
4. Add to appropriate crew configuration

## Troubleshooting

### Common Issues

**Import Errors**: Ensure you've installed in development mode (`pip install -e .`)

**API Rate Limits**: The system makes multiple API calls; consider rate limiting or using cached data

**Memory Issues**: Large SEC filings can be memory-intensive; adjust batch sizes in `utils/setup.py`

**Qdrant Connection**: Verify `QDRANT_CLUSTER_URL` and `QDRANT_API_KEY` are correct

**Missing Dependencies**: Run `pip install -e .` to ensure all dependencies are installed

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit your changes (`git commit -am 'Add new analysis crew'`)
4. Push to the branch (`git push origin feature/new-analysis`)
5. Create a Pull Request

## License

This project is private. All rights reserved.

## Acknowledgments

- Built with [CrewAI](https://github.com/joaomdmoura/crewAI)
- Financial data from [Yahoo Finance](https://finance.yahoo.com/)
- SEC filings via [sec-api](https://sec-api.io/)
- Vector search powered by [Qdrant](https://qdrant.tech/)
