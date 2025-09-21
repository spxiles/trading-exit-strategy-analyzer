# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a trading exit strategy analysis tool that backtests different profit-taking approaches using historical account balance data. The tool connects to a PostgreSQL database containing trading account data and analyzes the performance of various exit strategies.

## Core Architecture

### Main Components

1. **`main.py`** - Basic daily account balance summary export to CSV
2. **`exit_strategy_analysis.py`** - Comprehensive exit strategy backtesting engine

### Database Schema

The tool expects a PostgreSQL database with an `AccountBalance` table (case-sensitive, quoted identifier) containing:
- `timestamp` - Trade timestamp
- `liquidation_value` - Account value at that time
- `brokerage_account_id` - Account identifier

### Exit Strategy Analysis Engine

The core analysis (`exit_strategy_analysis.py`) implements:
- **Fixed Exit Strategies**: Tests exits at 1%, 2%, 3%, 5%, 10% profit thresholds
- **Optimal Exit Finder**: Iterates through 0.1% to 15% in 0.1% increments to find mathematically optimal exit percentage
- **Risk Metrics**: Calculates Sharpe ratio, volatility, maximum drawdown, and RoMaD (Return over Maximum Drawdown)
- **Compounded Returns**: Uses proper compounding rather than simple addition

## Common Commands

### Running Analysis
```bash
# Analyze all available data (default: last 30 days)
python3 exit_strategy_analysis.py

# Analyze specific lookback period
python3 exit_strategy_analysis.py 90  # Last 90 days
python3 exit_strategy_analysis.py 30  # Last 30 days
```

### Basic Export
```bash
# Export daily summary without strategy analysis
python3 main.py
```

### Dependencies
```bash
# Install dependencies
uv sync
```

## Database Connection

The tools expect these environment variables:
- `POSTGRES_HOST` (default: localhost)
- `POSTGRES_PORT` (default: 5433)
- `POSTGRES_DB` (default: looptrader)
- `POSTGRES_USER` (default: postgres)
- `POSTGRES_PASSWORD` (default: yourpassword)

## Data Filtering Rules

The analysis automatically applies these filters:
- **Date Range**: Only includes data from 2025-01-01 onwards
- **Weekend Exclusion**: Filters out Saturday (6) and Sunday (0) using PostgreSQL DOW
- **Outlier Removal**: Excludes days with >20% absolute close returns to remove data anomalies
- **Weekday Only**: Only analyzes trading days (Monday-Friday)

## Key Implementation Details

### Decimal Handling
Database returns `decimal.Decimal` values which must be converted to `float` before mathematical operations to avoid type errors.

### SQL Query Structure
Uses CTE (Common Table Expression) pattern with aggregated daily statistics, then calculates exit scenarios using CASE statements for different thresholds.

### Performance Metrics Calculation
- **Volatility**: Standard deviation of daily returns
- **Sharpe Ratio**: Average daily return divided by volatility
- **Maximum Drawdown**: Largest peak-to-trough decline calculated from cumulative returns
- **RoMaD**: Total return divided by maximum drawdown

### Output Files
Analysis generates timestamped CSV files:
- `exit_strategy_analysis_YYYYMMDD_HHMMSS.csv` - Detailed daily data
- `strategy_performance_summary_YYYYMMDD_HHMMSS.csv` - Strategy comparison metrics

## Terminal Output Order
Strategies display in logical progression:
1. EXIT AT 1PCT through 10PCT
2. HOLD TO CLOSE (baseline)
3. OPTIMAL EXIT (mathematically best percentage)