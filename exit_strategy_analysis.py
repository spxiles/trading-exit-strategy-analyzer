#!/usr/bin/env python3
"""
Exit Strategy Analysis: Compare hold-to-close vs early exit at various percentage thresholds.
"""

import csv
import os
from datetime import datetime, date, timedelta
from typing import List, Tuple, Dict, Set
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import CustomBusinessDay

import psycopg2


def get_market_holidays(start_year: int, end_year: int) -> Set[date]:
    """
    Get a set of US market holidays for the given year range.
    
    This includes:
    - New Year's Day
    - Martin Luther King Jr. Day
    - Presidents' Day
    - Good Friday
    - Memorial Day
    - Independence Day
    - Labor Day
    - Thanksgiving Day
    - Christmas Day
    
    Plus special closures and observed holidays when they fall on weekends.
    """
    cal = USFederalHolidayCalendar()
    
    # Get all federal holidays in range
    holidays = cal.holidays(start=f"{start_year}-01-01", end=f"{end_year}-12-31").date
    
    # Additional market-specific holidays like Good Friday
    # and special closures are not included in USFederalHolidayCalendar
    
    # Convert to set for faster lookups
    holiday_set = set(holidays)
    
    return holiday_set


# Define specific excluded trading days
EXCLUDED_TRADING_DAYS = {
    date(2025, 4, 30),  # Specific exclusion requested
    date(2025, 4, 2),   # Specific exclusion requested
    date(2025, 8, 21),  # Specific exclusion requested
    date(2025, 4, 10),  # Specific exclusion requested
    date(2025, 4, 9),  # Specific exclusion requested
}


def is_trading_day(day: date) -> bool:
    """
    Check if a given date is a trading day (not a weekend, holiday, or specifically excluded).
    
    Args:
        day: The date to check
        
    Returns:
        bool: True if it's a trading day, False otherwise
    """
    # Weekend check (5=Saturday, 6=Sunday)
    if day.weekday() >= 5:
        return False
    
    # Check if it's a specifically excluded date
    if day in EXCLUDED_TRADING_DAYS:
        return False
    
    # Get year range for holiday check
    start_year = day.year - 1  # Include previous year for observed holidays
    end_year = day.year + 1    # Include next year for observed holidays
    
    # Check if it's a holiday
    holidays = get_market_holidays(start_year, end_year)
    if day in holidays:
        return False
    
    return True


def get_daily_data_from_csv(brokerage_account_id: int = 85282889, csv_file: str = 'account_balance.csv', 
                      start_date: str = '2025-01-01', lookback_days: int | None = None):
    """Get daily data from CSV file for analysis."""
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        
        # Filter for the specific account
        account_df = df[df['accountNum'].astype(str) == str(brokerage_account_id)].copy()
        
        if account_df.empty:
            print(f"No data found for account {brokerage_account_id}")
            return []
            
        # Convert date to datetime
        account_df['date'] = pd.to_datetime(account_df['date']).dt.date
        
        # Filter by start date if provided
        if start_date:
            start_date = pd.to_datetime(start_date).date()
            account_df = account_df[account_df['date'] >= start_date]
            
        # Apply lookback days filter if specified
        if lookback_days:
            # Get the latest date in the dataset
            latest_date = account_df['date'].max()
            # Calculate the lookback cutoff date
            cutoff_date = latest_date - pd.Timedelta(days=lookback_days)
            # Filter for dates within the lookback period
            account_df = account_df[account_df['date'] >= cutoff_date]
        
        # Group by date to get daily statistics
        daily_stats = []
        skipped_holidays = 0
        skipped_weekends = 0
        skipped_specific_days = 0
        skipped_specific_dates = []
        
        for day, day_data in account_df.groupby('date'):
            # Skip weekends, holidays, and specific excluded days
            if not is_trading_day(day):
                # Count the type of day being skipped for reporting
                if day.weekday() >= 5:
                    skipped_weekends += 1
                elif day in EXCLUDED_TRADING_DAYS:
                    skipped_specific_days += 1
                    skipped_specific_dates.append(day.isoformat())
                else:
                    skipped_holidays += 1
                continue
            
            # Convert adjusted_nlv to float
            day_data['adjusted_nlv'] = day_data['adjusted_nlv'].astype(float)
            
            # Get first, max, min, last values for the day
            open_value = day_data['adjusted_nlv'].iloc[0]
            max_value = day_data['adjusted_nlv'].max()
            min_value = day_data['adjusted_nlv'].min()
            close_value = day_data['adjusted_nlv'].iloc[-1]
            
            # Calculate percentage changes
            high_pct = round(((max_value / open_value) - 1) * 100, 2)
            low_pct = round(((min_value / open_value) - 1) * 100, 2)
            close_pct = round(((close_value / open_value) - 1) * 100, 2)
            
            # Filter out extreme movements (same as SQL query)
            if abs(close_pct) > 20:
                continue
                
            # Calculate what return you'd get if you exited at various thresholds
            # Fine-grained exit strategies from 0.4% to 4% with 0.2% increments
            exit_at_0_4pct = 0.4 if high_pct >= 0.4 else close_pct
            exit_at_0_6pct = 0.6 if high_pct >= 0.6 else close_pct
            exit_at_0_8pct = 0.8 if high_pct >= 0.8 else close_pct
            exit_at_1pct = 1.0 if high_pct >= 1.0 else close_pct
            exit_at_1_2pct = 1.2 if high_pct >= 1.2 else close_pct
            exit_at_1_4pct = 1.4 if high_pct >= 1.4 else close_pct
            exit_at_1_6pct = 1.6 if high_pct >= 1.6 else close_pct
            exit_at_1_8pct = 1.8 if high_pct >= 1.8 else close_pct
            exit_at_2pct = 2.0 if high_pct >= 2.0 else close_pct
            exit_at_2_2pct = 2.2 if high_pct >= 2.2 else close_pct
            exit_at_2_4pct = 2.4 if high_pct >= 2.4 else close_pct
            exit_at_2_6pct = 2.6 if high_pct >= 2.6 else close_pct
            exit_at_2_8pct = 2.8 if high_pct >= 2.8 else close_pct
            exit_at_3pct = 3.0 if high_pct >= 3.0 else close_pct
            exit_at_3_2pct = 3.2 if high_pct >= 3.2 else close_pct
            exit_at_3_4pct = 3.4 if high_pct >= 3.4 else close_pct
            exit_at_3_6pct = 3.6 if high_pct >= 3.6 else close_pct
            exit_at_3_8pct = 3.8 if high_pct >= 3.8 else close_pct
            exit_at_4pct = 4.0 if high_pct >= 4.0 else close_pct
            # Original coarser thresholds
            exit_at_5pct = 5.0 if high_pct >= 5.0 else close_pct
            exit_at_10pct = 10.0 if high_pct >= 10.0 else close_pct
            
            daily_stats.append((
                day,
                open_value,
                max_value,
                min_value,
                close_value,
                high_pct,
                low_pct,
                close_pct,
                exit_at_0_4pct,
                exit_at_0_6pct,
                exit_at_0_8pct,
                exit_at_1pct,
                exit_at_1_2pct,
                exit_at_1_4pct,
                exit_at_1_6pct,
                exit_at_1_8pct,
                exit_at_2pct,
                exit_at_2_2pct,
                exit_at_2_4pct,
                exit_at_2_6pct,
                exit_at_2_8pct,
                exit_at_3pct,
                exit_at_3_2pct,
                exit_at_3_4pct,
                exit_at_3_6pct,
                exit_at_3_8pct,
                exit_at_4pct,
                exit_at_5pct,
                exit_at_10pct,
                high_pct  # daily_high_pct
            ))
        
        # Sort by date
        daily_stats.sort(key=lambda x: x[0])
        
        # Report on filtered days
        if skipped_holidays > 0 or skipped_weekends > 0 or skipped_specific_days > 0:
            print(f"Filtered out non-trading days: {skipped_weekends} weekends, {skipped_holidays} holidays, "
                  f"{skipped_specific_days} specifically excluded days")
            
            if skipped_specific_days > 0:
                print(f"Specifically excluded dates: {', '.join(skipped_specific_dates)}")
            
        return daily_stats
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return []


def get_daily_data_with_exit_analysis(brokerage_account_id: int = 1, lookback_days: int | None = None, use_csv: bool = False, csv_file: str = 'account_balance.csv'):
    """Get daily data and calculate optimal exit points."""
    
    # If using CSV, call the CSV function
    if use_csv:
        return get_daily_data_from_csv(
            brokerage_account_id=brokerage_account_id, 
            csv_file=csv_file,
            lookback_days=lookback_days
        )
    
    # Otherwise use database query
    db_params = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5433"),
        "database": os.getenv("POSTGRES_DB", "looptrader"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "yourpassword"),
    }

    # Get the base daily data plus calculate exit returns for different thresholds
    lookback_filter = ""
    query_params = [brokerage_account_id]

    if lookback_days:
        lookback_filter = "AND DATE(timestamp) >= CURRENT_DATE - INTERVAL '%s days'"
        query_params.append(lookback_days)

    query = f"""
    WITH daily_stats AS (
        SELECT
            DATE(timestamp) as date,
            (ARRAY_AGG(liquidation_value ORDER BY timestamp ASC))[1] as open_value,
            MAX(liquidation_value) as max_value,
            MIN(liquidation_value) as min_value,
            (ARRAY_AGG(liquidation_value ORDER BY timestamp DESC))[1] as close_value,
            ROUND(((MAX(liquidation_value) / (ARRAY_AGG(liquidation_value ORDER BY timestamp ASC))[1] - 1) * 100)::numeric, 2) as high_pct,
            ROUND(((MIN(liquidation_value) / (ARRAY_AGG(liquidation_value ORDER BY timestamp ASC))[1] - 1) * 100)::numeric, 2) as low_pct,
            ROUND((((ARRAY_AGG(liquidation_value ORDER BY timestamp DESC))[1] / (ARRAY_AGG(liquidation_value ORDER BY timestamp ASC))[1] - 1) * 100)::numeric, 2) as close_pct
        FROM public."AccountBalance"
        WHERE brokerage_account_id = %s
        AND DATE(timestamp) >= '2025-01-01' {lookback_filter}
        GROUP BY DATE(timestamp)
        HAVING ABS(ROUND((((ARRAY_AGG(liquidation_value ORDER BY timestamp DESC))[1] / (ARRAY_AGG(liquidation_value ORDER BY timestamp ASC))[1] - 1) * 100)::numeric, 2)) <= 20
    )
    SELECT
        date,
        open_value,
        max_value,
        min_value,
        close_value,
        high_pct,
        low_pct,
        close_pct,
        -- Calculate what return you'd get if you exited at various thresholds
        -- Fine-grained exit strategies from 0.4% to 4% with 0.2% increments
        CASE WHEN high_pct >= 0.4 THEN 0.4 ELSE close_pct END as exit_at_0_4pct,
        CASE WHEN high_pct >= 0.6 THEN 0.6 ELSE close_pct END as exit_at_0_6pct,
        CASE WHEN high_pct >= 0.8 THEN 0.8 ELSE close_pct END as exit_at_0_8pct,
        CASE WHEN high_pct >= 1.0 THEN 1.0 ELSE close_pct END as exit_at_1pct,
        CASE WHEN high_pct >= 1.2 THEN 1.2 ELSE close_pct END as exit_at_1_2pct,
        CASE WHEN high_pct >= 1.4 THEN 1.4 ELSE close_pct END as exit_at_1_4pct,
        CASE WHEN high_pct >= 1.6 THEN 1.6 ELSE close_pct END as exit_at_1_6pct,
        CASE WHEN high_pct >= 1.8 THEN 1.8 ELSE close_pct END as exit_at_1_8pct,
        CASE WHEN high_pct >= 2.0 THEN 2.0 ELSE close_pct END as exit_at_2pct,
        CASE WHEN high_pct >= 2.2 THEN 2.2 ELSE close_pct END as exit_at_2_2pct,
        CASE WHEN high_pct >= 2.4 THEN 2.4 ELSE close_pct END as exit_at_2_4pct,
        CASE WHEN high_pct >= 2.6 THEN 2.6 ELSE close_pct END as exit_at_2_6pct,
        CASE WHEN high_pct >= 2.8 THEN 2.8 ELSE close_pct END as exit_at_2_8pct,
        CASE WHEN high_pct >= 3.0 THEN 3.0 ELSE close_pct END as exit_at_3pct,
        CASE WHEN high_pct >= 3.2 THEN 3.2 ELSE close_pct END as exit_at_3_2pct,
        CASE WHEN high_pct >= 3.4 THEN 3.4 ELSE close_pct END as exit_at_3_4pct,
        CASE WHEN high_pct >= 3.6 THEN 3.6 ELSE close_pct END as exit_at_3_6pct,
        CASE WHEN high_pct >= 3.8 THEN 3.8 ELSE close_pct END as exit_at_3_8pct,
        CASE WHEN high_pct >= 4.0 THEN 4.0 ELSE close_pct END as exit_at_4pct,
        -- Original coarser thresholds
        CASE WHEN high_pct >= 5.0 THEN 5.0 ELSE close_pct END as exit_at_5pct,
        CASE WHEN high_pct >= 10.0 THEN 10.0 ELSE close_pct END as exit_at_10pct,
        high_pct as daily_high_pct
    FROM daily_stats
    ORDER BY date;
    """

    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, query_params)
                return cursor.fetchall()
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return []


def calculate_strategy_performance(data: List[Tuple]) -> dict:
    """Calculate cumulative performance for each exit strategy."""

    if not data:
        return {}

    strategies = {
        "hold_to_close": [],
        "exit_at_0_4pct": [],
        "exit_at_0_6pct": [],
        "exit_at_0_8pct": [],
        "exit_at_1pct": [],
        "exit_at_1_2pct": [],
        "exit_at_1_4pct": [],
        "exit_at_1_6pct": [],
        "exit_at_1_8pct": [],
        "exit_at_2pct": [],
        "exit_at_2_2pct": [],
        "exit_at_2_4pct": [],
        "exit_at_2_6pct": [],
        "exit_at_2_8pct": [],
        "exit_at_3pct": [],
        "exit_at_3_2pct": [],
        "exit_at_3_4pct": [],
        "exit_at_3_6pct": [],
        "exit_at_3_8pct": [],
        "exit_at_4pct": [],
        "exit_at_5pct": [],
        "exit_at_10pct": [],
    }

    # Collect all daily highs to find optimal exit points
    daily_highs = []

    for row in data:
        # Unpack the row data with all the new exit strategies
        # The first 8 items are: date, open, max, min, close, high_pct, low_pct, close_pct
        date_val, open_val, max_val, min_val, close_val, high_pct, low_pct, close_pct = row[:8]
        
        # The next items are all the exit strategy percentages, in order
        exit_0_4, exit_0_6, exit_0_8 = row[8:11]
        exit_1, exit_1_2, exit_1_4, exit_1_6, exit_1_8 = row[11:16]
        exit_2, exit_2_2, exit_2_4, exit_2_6, exit_2_8 = row[16:21]
        exit_3, exit_3_2, exit_3_4, exit_3_6, exit_3_8 = row[21:26]
        exit_4, exit_5, exit_10 = row[26:29]
        daily_high = row[29]  # Last item is daily_high_pct
        
        # Add to each strategy
        strategies["hold_to_close"].append(close_pct)
        strategies["exit_at_0_4pct"].append(exit_0_4)
        strategies["exit_at_0_6pct"].append(exit_0_6)
        strategies["exit_at_0_8pct"].append(exit_0_8)
        strategies["exit_at_1pct"].append(exit_1)
        strategies["exit_at_1_2pct"].append(exit_1_2)
        strategies["exit_at_1_4pct"].append(exit_1_4)
        strategies["exit_at_1_6pct"].append(exit_1_6)
        strategies["exit_at_1_8pct"].append(exit_1_8)
        strategies["exit_at_2pct"].append(exit_2)
        strategies["exit_at_2_2pct"].append(exit_2_2)
        strategies["exit_at_2_4pct"].append(exit_2_4)
        strategies["exit_at_2_6pct"].append(exit_2_6)
        strategies["exit_at_2_8pct"].append(exit_2_8)
        strategies["exit_at_3pct"].append(exit_3)
        strategies["exit_at_3_2pct"].append(exit_3_2)
        strategies["exit_at_3_4pct"].append(exit_3_4)
        strategies["exit_at_3_6pct"].append(exit_3_6)
        strategies["exit_at_3_8pct"].append(exit_3_8)
        strategies["exit_at_4pct"].append(exit_4)
        strategies["exit_at_5pct"].append(exit_5)
        strategies["exit_at_10pct"].append(exit_10)
        
        daily_highs.append(float(daily_high))

    # Calculate summary statistics
    results = {}
    for strategy_name, returns in strategies.items():
        # Calculate compounded return: (1 + r1/100) * (1 + r2/100) * ... - 1
        compounded_return = 1.0
        for daily_return in returns:
            compounded_return *= 1 + float(daily_return) / 100
        total_return_pct = (compounded_return - 1) * 100

        # Convert all returns to float to avoid decimal.Decimal issues
        float_returns = [float(r) for r in returns]

        avg_daily_return = sum(float_returns) / len(float_returns) if float_returns else 0
        winning_days = len([r for r in float_returns if r > 0])
        losing_days = len([r for r in float_returns if r < 0])
        win_rate = winning_days / len(float_returns) if float_returns else 0

        # Calculate volatility (standard deviation of daily returns)
        if len(float_returns) > 1:
            variance = sum((r - avg_daily_return) ** 2 for r in float_returns) / (len(float_returns) - 1)
            volatility = variance**0.5
        else:
            volatility = 0

        # Calculate maximum drawdown and RoMaD
        max_drawdown, romad = calculate_drawdown_metrics(float_returns)

        # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0

        results[strategy_name] = {
            "total_return": round(total_return_pct, 2),
            "avg_daily_return": round(avg_daily_return, 2),
            "volatility": round(volatility, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown, 2),
            "romad": round(romad, 2),
            "winning_days": winning_days,
            "losing_days": losing_days,
            "total_days": len(float_returns),
            "win_rate": round(win_rate * 100, 2),
            "best_day": round(max(float_returns), 2) if float_returns else 0,
            "worst_day": round(min(float_returns), 2) if float_returns else 0,
        }

    # Find optimal exit percentage by testing different thresholds
    optimal_exit = find_optimal_exit_percentage(data)
    results["optimal_exit"] = optimal_exit

    return results


def calculate_drawdown_metrics(returns: List[float]) -> Tuple[float, float]:
    """Calculate maximum drawdown and RoMaD (Return over Maximum Drawdown)."""
    if not returns:
        return 0.0, 0.0

    # Calculate cumulative returns
    cumulative_value = 1.0
    cumulative_values = [cumulative_value]

    for daily_return in returns:
        cumulative_value *= 1 + float(daily_return) / 100
        cumulative_values.append(cumulative_value)

    # Calculate running maximum and drawdowns
    running_max = cumulative_values[0]
    max_drawdown = 0.0

    for value in cumulative_values:
        if value > running_max:
            running_max = value

        drawdown = (running_max - value) / running_max * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Calculate total return percentage
    total_return_pct = (cumulative_values[-1] - 1) * 100

    # Calculate RoMaD (Return over Maximum Drawdown)
    romad = total_return_pct / max_drawdown if max_drawdown > 0 else float("inf")

    return max_drawdown, romad


def find_optimal_exit_percentage(data: List[Tuple]) -> dict:
    """Find the optimal exit percentage by testing various thresholds."""

    # Generate test percentages: 0.1% to 15% in 0.1% increments
    test_percentages = [round(x * 0.1, 1) for x in range(1, 151)]

    best_return = float("-inf")
    best_percentage = 0

    for test_pct in test_percentages:
        # Calculate what the total return would be at this exit percentage
        total_return = 1.0

        for row in data:
            # Get the daily high (last item) and close_pct (8th item, index 7)
            daily_high = float(row[29])  # daily_high_pct is the last item
            close_pct = float(row[7])    # close_pct is the 8th item

            # If daily high reached our test percentage, exit there; otherwise use close
            exit_return = test_pct if daily_high >= test_pct else close_pct
            total_return *= 1 + exit_return / 100

        final_return_pct = (total_return - 1) * 100

        if final_return_pct > best_return:
            best_return = final_return_pct
            best_percentage = test_pct

    # Calculate stats for the optimal strategy
    optimal_returns = []
    for row in data:
        # Get the daily high (last item) and close_pct (8th item, index 7)
        daily_high = float(row[29])  # daily_high_pct is the last item
        close_pct = float(row[7])    # close_pct is the 8th item

        exit_return = best_percentage if daily_high >= best_percentage else close_pct
        optimal_returns.append(exit_return)

    # Convert optimal returns to float to avoid decimal issues
    float_optimal_returns = [float(r) for r in optimal_returns]

    winning_days = len([r for r in float_optimal_returns if r > 0])
    losing_days = len([r for r in float_optimal_returns if r < 0])
    win_rate = winning_days / len(float_optimal_returns) if float_optimal_returns else 0
    avg_daily_return = sum(float_optimal_returns) / len(float_optimal_returns) if float_optimal_returns else 0

    # Calculate volatility for optimal strategy
    if len(float_optimal_returns) > 1:
        variance = sum((r - avg_daily_return) ** 2 for r in float_optimal_returns) / (len(float_optimal_returns) - 1)
        volatility = variance**0.5
    else:
        volatility = 0

    # Calculate drawdown metrics for optimal strategy
    max_drawdown, romad = calculate_drawdown_metrics(float_optimal_returns)

    # Calculate Sharpe ratio for optimal strategy
    sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0

    return {
        "exit_percentage": best_percentage,
        "total_return": round(best_return, 2),
        "avg_daily_return": round(avg_daily_return, 2),
        "volatility": round(volatility, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown": round(max_drawdown, 2),
        "romad": round(romad, 2),
        "winning_days": winning_days,
        "losing_days": losing_days,
        "total_days": len(float_optimal_returns),
        "win_rate": round(win_rate * 100, 2),
        "best_day": round(max(float_optimal_returns), 2) if float_optimal_returns else 0,
        "worst_day": round(min(float_optimal_returns), 2) if float_optimal_returns else 0,
    }


def export_detailed_analysis(data: List[Tuple], filename: str | None = None):
    """Export detailed day-by-day analysis."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exit_strategy_analysis_{timestamp}.csv"

    headers = [
        "date",
        "open_value",
        "max_value",
        "min_value",
        "close_value",
        "high_pct",
        "low_pct",
        "close_pct",
        "exit_at_0_4pct",
        "exit_at_0_6pct",
        "exit_at_0_8pct",
        "exit_at_1pct",
        "exit_at_1_2pct",
        "exit_at_1_4pct",
        "exit_at_1_6pct",
        "exit_at_1_8pct",
        "exit_at_2pct",
        "exit_at_2_2pct",
        "exit_at_2_4pct",
        "exit_at_2_6pct",
        "exit_at_2_8pct",
        "exit_at_3pct",
        "exit_at_3_2pct",
        "exit_at_3_4pct",
        "exit_at_3_6pct",
        "exit_at_3_8pct",
        "exit_at_4pct",
        "exit_at_5pct",
        "exit_at_10pct",
        "daily_high_pct",
    ]

    try:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(data)
        print(f"Detailed analysis exported to {filename}")
        return filename
    except IOError as e:
        print(f"Error writing to file: {e}")
        return None
        
        
def export_daily_stats(daily_stats: List[Tuple], filename: str | None = None):
    """
    Export the raw daily statistics data to a CSV file.
    
    Args:
        daily_stats: The list of daily statistics tuples
        filename: Optional filename, will be auto-generated if not provided
        
    Returns:
        The filename if export was successful, None otherwise
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"daily_stats_{timestamp}.csv"

    headers = [
        "date",
        "open_value",
        "max_value",
        "min_value",
        "close_value",
        "high_pct",
        "low_pct",
        "close_pct",
        "exit_at_0_4pct",
        "exit_at_0_6pct",
        "exit_at_0_8pct",
        "exit_at_1pct",
        "exit_at_1_2pct",
        "exit_at_1_4pct",
        "exit_at_1_6pct",
        "exit_at_1_8pct",
        "exit_at_2pct",
        "exit_at_2_2pct",
        "exit_at_2_4pct",
        "exit_at_2_6pct",
        "exit_at_2_8pct",
        "exit_at_3pct",
        "exit_at_3_2pct",
        "exit_at_3_4pct",
        "exit_at_3_6pct",
        "exit_at_3_8pct",
        "exit_at_4pct",
        "exit_at_5pct",
        "exit_at_10pct",
        "daily_high_pct",
    ]

    try:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            # Format dates as strings for CSV output
            formatted_data = []
            for row in daily_stats:
                # Convert first item (date) to string if it's a date object
                if isinstance(row[0], date):
                    formatted_row = [row[0].isoformat()] + list(row[1:])
                else:
                    formatted_row = list(row)
                formatted_data.append(formatted_row)
                
            writer.writerows(formatted_data)
        print(f"Daily statistics exported to {filename}")
        return filename
    except IOError as e:
        print(f"Error writing to file: {e}")
        return None


def export_strategy_summary(performance: dict, filename: str | None = None):
    """Export strategy performance summary."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_performance_summary_{timestamp}.csv"

    headers = [
        "strategy",
        "total_return_pct",
        "avg_daily_return_pct",
        "volatility_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "romad",
        "winning_days",
        "losing_days",
        "total_days",
        "win_rate_pct",
        "best_day_pct",
        "worst_day_pct",
    ]

    try:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for strategy, stats in performance.items():
                if strategy == "optimal_exit":
                    strategy_name = f"{strategy} (at {stats['exit_percentage']}%)"
                else:
                    strategy_name = strategy

                row = [
                    strategy_name,
                    stats["total_return"],
                    stats["avg_daily_return"],
                    stats["volatility"],
                    stats["sharpe_ratio"],
                    stats["max_drawdown"],
                    stats["romad"],
                    stats["winning_days"],
                    stats["losing_days"],
                    stats["total_days"],
                    stats["win_rate"],
                    stats["best_day"],
                    stats["worst_day"],
                ]
                writer.writerow(row)

        print(f"Strategy summary exported to {filename}")
        return filename
    except IOError as e:
        print(f"Error writing to file: {e}")
        return None


def main():
    """Main function to run the exit strategy analysis."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Run exit strategy analysis')
    parser.add_argument('--days', type=int, default=30, 
                        help='Number of lookback days (default: 30)')
    parser.add_argument('--account', type=int, default=85282889, 
                        help='Brokerage account ID (default: 85282889)')
    parser.add_argument('--csv', action='store_true',
                        help='Use CSV file instead of database')
    parser.add_argument('--csv-file', type=str, default='account_balance.csv',
                        help='CSV file path (default: account_balance.csv)')
    
    # Handle both new argparse and old sys.argv for backward compatibility
    if len(sys.argv) > 1 and sys.argv[1].isdigit() and '--' not in sys.argv[1]:
        # Old style: python exit_strategy_analysis.py [days]
        try:
            lookback_days = int(sys.argv[1])
            print(f"Running exit strategy analysis for last {lookback_days} days...")
            account_id = 85282889
            use_csv = True
            csv_file = 'account_balance.csv'
        except ValueError:
            print("Invalid number of days. Usage: python exit_strategy_analysis.py [days]")
            return
    else:
        # New style with argparse
        args = parser.parse_args()
        lookback_days = args.days
        account_id = args.account
        use_csv = args.csv or True  # Default to using CSV
        csv_file = args.csv_file
        
        print(f"Running exit strategy analysis for account {account_id} using {'CSV' if use_csv else 'database'}")
        if lookback_days:
            print(f"Looking back {lookback_days} days")
        else:
            print("Analyzing all available data")

    # Get the data
    data = get_daily_data_with_exit_analysis(
        brokerage_account_id=account_id,
        lookback_days=lookback_days,
        use_csv=use_csv,
        csv_file=csv_file
    )

    if not data:
        print("No data found or error occurred.")
        return

    start_date = data[0][0]  # First date in results
    end_date = data[-1][0]  # Last date in results
    date_range_text = f"from {start_date} to {end_date}"
    period_text = f"last {lookback_days} days" if lookback_days else "all available data"
    print(f"Analyzing {len(data)} trading days ({period_text}, {date_range_text})")

    # Calculate performance for each strategy
    performance = calculate_strategy_performance(data)

    # Display results
    print("\n" + "=" * 80)
    print("EXIT STRATEGY PERFORMANCE COMPARISON")
    print("=" * 80)

    # Define the order for display
    strategy_order = [
        "exit_at_0_4pct",
        "exit_at_0_6pct",
        "exit_at_0_8pct",
        "exit_at_1pct",
        "exit_at_1_2pct",
        "exit_at_1_4pct",
        "exit_at_1_6pct",
        "exit_at_1_8pct",
        "exit_at_2pct",
        "exit_at_2_2pct",
        "exit_at_2_4pct",
        "exit_at_2_6pct",
        "exit_at_2_8pct",
        "exit_at_3pct",
        "exit_at_3_2pct",
        "exit_at_3_4pct",
        "exit_at_3_6pct",
        "exit_at_3_8pct",
        "exit_at_4pct",
        "exit_at_5pct",
        "exit_at_10pct",
        "hold_to_close",
        "optimal_exit",
    ]

    for strategy in strategy_order:
        if strategy in performance:
            stats = performance[strategy]
            if strategy == "optimal_exit":
                print(f"\n{strategy.upper().replace('_', ' ')} (at {stats['exit_percentage']}%):")
            else:
                print(f"\n{strategy.upper().replace('_', ' ')}:")
            print(f"  Total Return: {stats['total_return']:+.2f}%")
            print(f"  Avg Daily Return: {stats['avg_daily_return']:+.2f}%")
            print(f"  Volatility: {stats['volatility']:.2f}%")
            print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {stats['max_drawdown']:.2f}%")
            print(f"  RoMaD: {stats['romad']:.2f}")
            print(f"  Win Rate: {stats['win_rate']:.1f}% ({stats['winning_days']}/{stats['total_days']} days)")
            print(f"  Best/Worst Day: {stats['best_day']:+.2f}% / {stats['worst_day']:+.2f}%")

    # Find the best strategy
    best_strategy = max(performance.items(), key=lambda x: x[1]["total_return"])
    print(f"\n{'=' * 80}")
    print(f"BEST STRATEGY: {best_strategy[0].upper().replace('_', ' ')}")
    print(f"Total Return: {best_strategy[1]['total_return']:+.2f}%")
    print(
        f"Outperformed hold-to-close by: {best_strategy[1]['total_return'] - performance['hold_to_close']['total_return']:+.2f}%"
    )
    print("=" * 80)

    # Export files
    export_detailed_analysis(data)
    export_strategy_summary(performance)
    export_daily_stats(data)


if __name__ == "__main__":
    main()
