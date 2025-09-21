# Trading Exit Strategy Analyzer

A comprehensive backtesting tool for analyzing trading exit strategies with risk-adjusted performance metrics. This tool helps traders optimize profit-taking decisions by analyzing historical account data.

## =€ Features

- **Multiple Exit Strategy Testing**: Compare holding to close vs. exiting at 1%, 2%, 3%, 5%, and 10% profit thresholds
- **Optimal Exit Finder**: Automatically discovers the mathematically best exit percentage for your historical data
- **Risk-Adjusted Metrics**: Calculate Sharpe ratio, volatility, maximum drawdown, and RoMaD (Return over Maximum Drawdown)
- **Flexible Time Periods**: Analyze any lookback period (30, 60, 90 days, etc.)
- **Data Quality Controls**: Automatic filtering of weekends, outliers, and data anomalies
- **Comprehensive Reporting**: Detailed CSV exports and terminal summaries

## =Ê What It Analyzes

The tool answers critical trading questions:
- Should I hold positions to close or take profits early?
- What's the optimal profit-taking percentage for my strategy?
- How do different exit strategies perform on a risk-adjusted basis?
- What's my maximum drawdown and how does it compare across strategies?

## =à Setup

### Prerequisites
- Python 3.11+
- PostgreSQL database with trading account data
- UV package manager (recommended) or pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/spxiles/trading-exit-strategy-analyzer.git
cd trading-exit-strategy-analyzer
```

2. Install dependencies:
```bash
uv sync
```

3. Set up environment variables:
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5433
export POSTGRES_DB=your_database
export POSTGRES_USER=your_username
export POSTGRES_PASSWORD=your_password
```

### Database Requirements

Your PostgreSQL database should contain an `AccountBalance` table with:
- `timestamp` - Trade timestamp
- `liquidation_value` - Account value at that time
- `brokerage_account_id` - Account identifier

## = Usage

### Quick Start

Analyze the last 30 days:
```bash
python3 exit_strategy_analysis.py
```

### Custom Time Periods

```bash
# Last 90 days
python3 exit_strategy_analysis.py 90

# Last 7 days
python3 exit_strategy_analysis.py 7
```

### Basic Daily Summary

Export daily account balance summary without strategy analysis:
```bash
python3 main.py
```

## =È Sample Output

```
Running exit strategy analysis for last 30 days...
Analyzing 22 trading days (last 30 days, from 2025-01-15 to 2025-02-15)

================================================================================
EXIT STRATEGY PERFORMANCE COMPARISON
================================================================================

EXIT AT 1PCT:
  Total Return: +8.45%
  Avg Daily Return: +0.38%
  Volatility: 1.23%
  Sharpe Ratio: 0.31
  Max Drawdown: 2.15%
  RoMaD: 3.93
  Win Rate: 68.2% (15/22 days)
  Best/Worst Day: +1.00% / -1.85%

HOLD TO CLOSE:
  Total Return: +6.23%
  Avg Daily Return: +0.28%
  Volatility: 1.45%
  Sharpe Ratio: 0.19
  Max Drawdown: 3.42%
  RoMaD: 1.82
  Win Rate: 59.1% (13/22 days)
  Best/Worst Day: +2.15% / -2.89%

OPTIMAL EXIT (at 1.7%):
  Total Return: +9.12%
  Avg Daily Return: +0.41%
  Volatility: 1.28%
  Sharpe Ratio: 0.32
  Max Drawdown: 2.08%
  RoMaD: 4.38
  Win Rate: 72.7% (16/22 days)
  Best/Worst Day: +1.70% / -1.85%

================================================================================
BEST STRATEGY: OPTIMAL EXIT (AT 1.7%)
Total Return: +9.12%
Outperformed hold-to-close by: +2.89%
================================================================================
```

## =Á Output Files

The analysis generates timestamped CSV files:
- `exit_strategy_analysis_YYYYMMDD_HHMMSS.csv` - Day-by-day detailed analysis
- `strategy_performance_summary_YYYYMMDD_HHMMSS.csv` - Strategy comparison metrics

## <¯ Key Metrics Explained

- **Total Return**: Compounded return over the analysis period
- **Volatility**: Standard deviation of daily returns (risk measure)
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline
- **RoMaD**: Return over Maximum Drawdown (efficiency measure)
- **Win Rate**: Percentage of profitable trading days

## =' Configuration

The tool automatically filters data to ensure quality analysis:
- **Date Range**: Only analyzes data from 2025-01-01 onwards
- **Trading Days**: Excludes weekends automatically
- **Outlier Removal**: Filters out days with >20% absolute returns
- **Account Focus**: Analyzes brokerage_account_id = 1 by default

## > Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## =Ä License

This project is open source and available under the MIT License.

---

*Built with Claude Code - AI-powered development assistant*