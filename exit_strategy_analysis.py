#!/usr/bin/env python3
"""
Exit Strategy Analysis: Compare hold-to-close vs early exit at various percentage thresholds.
"""

import csv
import os
from datetime import datetime
from typing import List, Tuple

import psycopg2


def get_daily_data_with_exit_analysis(brokerage_account_id: int = 1, lookback_days: int | None = None):
    """Get daily data and calculate optimal exit points."""

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
        AND EXTRACT(DOW FROM DATE(timestamp)) NOT IN (0, 6)
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
        CASE
            WHEN high_pct >= 1.0 THEN 1.0
            ELSE close_pct
        END as exit_at_1pct,
        CASE
            WHEN high_pct >= 2.0 THEN 2.0
            ELSE close_pct
        END as exit_at_2pct,
        CASE
            WHEN high_pct >= 3.0 THEN 3.0
            ELSE close_pct
        END as exit_at_3pct,
        CASE
            WHEN high_pct >= 5.0 THEN 5.0
            ELSE close_pct
        END as exit_at_5pct,
        CASE
            WHEN high_pct >= 10.0 THEN 10.0
            ELSE close_pct
        END as exit_at_10pct,
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
        "exit_at_1pct": [],
        "exit_at_2pct": [],
        "exit_at_3pct": [],
        "exit_at_5pct": [],
        "exit_at_10pct": [],
    }

    # Collect all daily highs to find optimal exit points
    daily_highs = []

    for row in data:
        _, _, _, _, _, _, _, close_pct, exit_1, exit_2, exit_3, exit_5, exit_10, daily_high = row

        strategies["hold_to_close"].append(close_pct)
        strategies["exit_at_1pct"].append(exit_1)
        strategies["exit_at_2pct"].append(exit_2)
        strategies["exit_at_3pct"].append(exit_3)
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
            _, _, _, _, _, _, _, close_pct, _, _, _, _, _, daily_high = row
            daily_high = float(daily_high)
            close_pct = float(close_pct)

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
        _, _, _, _, _, _, _, close_pct, _, _, _, _, _, daily_high = row
        daily_high = float(daily_high)
        close_pct = float(close_pct)

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
        "exit_at_1pct",
        "exit_at_2pct",
        "exit_at_3pct",
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

    # Parse command line arguments for lookback days
    lookback_days = None
    if len(sys.argv) > 1:
        try:
            lookback_days = int(sys.argv[1])
            print(f"Running exit strategy analysis for last {lookback_days} days...")
        except ValueError:
            print("Invalid number of days. Usage: python exit_strategy_analysis.py [days]")
            return
    else:
        lookback_days = 30
        print("Running exit strategy analysis for all available data...")

    # Get the data
    data = get_daily_data_with_exit_analysis(lookback_days=lookback_days)

    if not data:
        print("No data found or database error occurred.")
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
        "exit_at_1pct",
        "exit_at_2pct",
        "exit_at_3pct",
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


if __name__ == "__main__":
    main()
