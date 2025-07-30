import matplotlib.pyplot as plt
import yfinance as yf

# visualize the stock line of a particular ticker and timeline
def visualize(ticker, start, end):
    """
    Fetches and visualizes the closing price of a stock over a given time period.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA').
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
    """
    # Download historical data
    data = yf.download(ticker, start=start, end=end)

    # Check if data was returned
    if data.empty:
        print(f"No data found for {ticker} between {start} and {end}.")
        return

    # Plot the closing price
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.title(f"{ticker} Stock Price from {start} to {end}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize('AMD', '2018-01-01', '2023-12-31')