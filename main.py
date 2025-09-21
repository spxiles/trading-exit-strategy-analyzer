#!/usr/bin/env python3
"""
Export daily account balance summary to CSV.
Gets first, max, min, and last liquidation_value for each day from AccountBalance table.
"""

import csv
import os
from datetime import datetime

import psycopg2


def get_daily_liquidation_summary(brokerage_account_id: int = 1):
    """Query daily liquidation value summary from AccountBalance table."""

    # Database connection parameters from environment
    db_params = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5433"),
        "database": os.getenv("POSTGRES_DB", "looptrader"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "yourpassword"),
    }

    query = """
    SELECT
        DATE(timestamp) as date,
        (ARRAY_AGG(liquidation_value ORDER BY timestamp ASC))[1] as first_liquidation_value,
        MAX(liquidation_value) as max_liquidation_value,
        MIN(liquidation_value) as min_liquidation_value,
        (ARRAY_AGG(liquidation_value ORDER BY timestamp DESC))[1] as last_liquidation_value,
        ROUND(((MAX(liquidation_value) / (ARRAY_AGG(liquidation_value ORDER BY timestamp ASC))[1] - 1) * 100)::numeric, 2) as high_pct_of_open,
        ROUND(((MIN(liquidation_value) / (ARRAY_AGG(liquidation_value ORDER BY timestamp ASC))[1] - 1) * 100)::numeric, 2) as low_pct_of_open,
        ROUND((((ARRAY_AGG(liquidation_value ORDER BY timestamp DESC))[1] / (ARRAY_AGG(liquidation_value ORDER BY timestamp ASC))[1] - 1) * 100)::numeric, 2) as close_pct_of_open
    FROM public."AccountBalance"
    WHERE brokerage_account_id = %s
    GROUP BY DATE(timestamp)
    HAVING ABS(ROUND((((ARRAY_AGG(liquidation_value ORDER BY timestamp DESC))[1] / (ARRAY_AGG(liquidation_value ORDER BY timestamp ASC))[1] - 1) * 100)::numeric, 2)) <= 20
    ORDER BY date;
    """

    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (brokerage_account_id,))
                return cursor.fetchall()
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return []


def export_to_csv(data, filename: str | None = None):
    """Export query results to CSV file."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"account_balance_daily_summary_{timestamp}.csv"

    headers = [
        "date",
        "first_liquidation_value",
        "max_liquidation_value",
        "min_liquidation_value",
        "last_liquidation_value",
        "high_pct_of_open",
        "low_pct_of_open",
        "close_pct_of_open",
    ]

    try:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(data)
        print(f"Data exported to {filename}")
        return filename
    except IOError as e:
        print(f"Error writing to file: {e}")
        return None


def main():
    """Main function to execute the export."""
    print("Fetching daily account balance summary...")
    data = get_daily_liquidation_summary()

    if not data:
        print("No data found or database error occurred.")
        return

    print(f"Found {len(data)} days of data")
    export_to_csv(data)


if __name__ == "__main__":
    main()
