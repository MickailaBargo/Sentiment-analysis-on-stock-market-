from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO
import base64
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize Flask app
app = Flask(__name__)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()


# Function to fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    # Fetch stock data using yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)  # Reset index to make Date a column

    # Debugging: Print available columns
    print("Columns in stock_data BEFORE flattening:", stock_data.columns)

    # Flatten MultiIndex if it exists
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = ['_'.join(filter(None, col)) for col in stock_data.columns]

    # Debugging: Print available columns after flattening
    print("Columns in stock_data AFTER flattening:", stock_data.columns)

    # Dynamically identify the 'Close' column
    close_column = [col for col in stock_data.columns if 'Close' in col]
    if not close_column:
        raise KeyError("The 'Close' column is not found in the downloaded stock data.")

    close_column = close_column[0]  # Take the first match
    stock_data = stock_data[['Date', close_column]]
    stock_data.rename(columns={close_column: 'Close'}, inplace=True)  # Rename it to 'Close'

    return stock_data


# Function to fetch social media data (placeholder)
def fetch_social_media_data(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date)
    sentiments = np.random.uniform(-1, 1, len(dates))  # Random sentiments
    return pd.DataFrame({'Date': dates, 'Sentiment': sentiments})


# Function to validate and prepare dataframe
def validate_and_prepare_dataframe(df, date_column):
    # Ensure the date_column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Ensure the date_column is not part of the index
    if date_column not in df.columns:
        df = df.reset_index()

    # Drop duplicate indices, if any
    df = df.reset_index(drop=True)
    return df


@app.route("/")
def home():
    # Define date range
    end_date = datetime(2024, 11, 1)
    start_date = end_date - timedelta(days=365)

    # Fetch stock data and social media data
    try:
        stock_data = fetch_stock_data('ABNB', start_date, end_date)
    except KeyError as e:
        return f"Error in fetching stock data: {e}"

    social_media_data = fetch_social_media_data(start_date, end_date)

    # Validate and prepare both dataframes
    stock_data = validate_and_prepare_dataframe(stock_data, 'Date')
    social_media_data = validate_and_prepare_dataframe(social_media_data, 'Date')

    # Debugging: Check the structure of both dataframes
    print("Stock Data Structure:")
    print(stock_data.head())
    print(stock_data.info())
    print("Social Media Data Structure:")
    print(social_media_data.head())
    print(social_media_data.info())

    # Merge datasets on Date
    try:
        merged_data = pd.merge(stock_data, social_media_data, on='Date', how='inner')
    except Exception as e:
        return f"Error during merging: {e}. <br> Stock Data Columns: {stock_data.columns} <br> Social Media Data Columns: {social_media_data.columns}"

    # Calculate daily stock returns
    merged_data['Return'] = merged_data['Close'].pct_change()

    # Drop NaN values
    merged_data.dropna(inplace=True)

    # Calculate correlation
    correlation = merged_data['Sentiment'].corr(merged_data['Return'])

    # Generate plot
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

    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Return HTML response with plot and correlation
    return render_template('index.html', plot_url=plot_url, correlation=correlation)


if __name__ == "__main__":
    app.run(debug=True)
