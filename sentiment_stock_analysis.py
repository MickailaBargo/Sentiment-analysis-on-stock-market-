# File: sentiment_stock_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)  # Reset index to make Date a column
    return stock_data[['Date', 'Close']]

# Function to fetch social media data (placeholder)
def fetch_social_media_data(start_date, end_date):
    # Placeholder function: Replace with actual data fetching logic
    dates = pd.date_range(start=start_date, end=end_date)
    sentiments = np.random.uniform(-1, 1, len(dates))  # Random sentiments as placeholder
    return pd.DataFrame({'Date': dates, 'Sentiment': sentiments})

# Main function
def main():
    # Define date range
    end_date = datetime(2024, 11, 1)  # Up to 2024-11-01
    start_date = end_date - timedelta(days=365)  # Past year from end_date

    # Fetch stock data
    stock_data = fetch_stock_data('ABNB', start_date, end_date)

    # Fetch social media data
    social_media_data = fetch_social_media_data(start_date, end_date)

    # Ensure Date columns are properly formatted
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    social_media_data['Date'] = pd.to_datetime(social_media_data['Date'])

    # Reset indices and ensure Date columns are at the same level
    stock_data = stock_data.reset_index(drop=True)
    social_media_data = social_media_data.reset_index(drop=True)

    # Check if Date is part of the index; if so, reset it
    if 'Date' not in stock_data.columns:
        stock_data.reset_index(inplace=True)

    if 'Date' not in social_media_data.columns:
        social_media_data.reset_index(inplace=True)

    # Ensure both dataframes have single-level Date columns
    if isinstance(stock_data.index, pd.MultiIndex):
        stock_data = stock_data.reset_index(drop=True)

    if isinstance(social_media_data.index, pd.MultiIndex):
        social_media_data = social_media_data.reset_index(drop=True)

    # Merge datasets on Date
    try:
        merged_data = pd.merge(stock_data, social_media_data, on='Date', how='inner')
    except Exception as e:
        print(f"Error during merging: {e}")
        print("Inspecting stock_data and social_media_data structures:")
        print(stock_data.head())
        print(stock_data.info())
        print(social_media_data.head())
        print(social_media_data.info())
        return

    # Calculate daily stock returns
    merged_data['Return'] = merged_data['Close'].pct_change()

    # Drop NaN values
    merged_data.dropna(inplace=True)

    # Calculate correlation
    correlation = merged_data['Sentiment'].corr(merged_data['Return'])
    print(f'Correlation between sentiment and stock returns: {correlation:.2f}')

    # Plot data
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Close Price', color='tab:blue')
    ax1.plot(merged_data['Date'], merged_data['Close'], color='tab:blue', label='Stock Close Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Sentiment Score', color='tab:red')
    ax2.plot(merged_data['Date'], merged_data['Sentiment'], color='tab:red', label='Sentiment Score')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title('Airbnb Stock Price and Social Media Sentiment (2023-2024)')
    plt.show()

if __name__ == '__main__':
    main()
