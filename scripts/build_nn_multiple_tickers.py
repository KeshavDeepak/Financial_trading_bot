import json
from datetime import datetime
from backend.utils import (
    backtest_multiple_tickers,
    plot_training_history,
    plot_portfolio_timeline,
    plot_backtest_results,
    plot_trade_signals,
    print_trade_summary,
)
from backend.lstm_nn_4 import LSTMTradingAgent

if __name__ == "__main__":
    # Initialize the trading agent
    agent = LSTMTradingAgent(look_back=60, train_test_split=0.8, embedding_dim=8)

    # Choose tickers and timeline
    tickers = ["AAPL", "MSFT", "AMD", "TSLA", "GOOG"]
    start_date, end_date = "2018-01-01", "2023-12-31"
    initial_balance = 10000
    interval = "1d"

    # Hyperparameters
    epochs = 5
    batch_size = 32

    try:
        # Fetch and prepare data
        all_data = agent.fetch_data(tickers, start_date, end_date, interval)
        agent.prepare_data(all_data)

        # Build and train model
        agent.build_model()
        agent.train_model(epochs=epochs, batch_size=batch_size)
        
        # Save the model and scaler
        current_time = datetime.now().strftime("%d %b - %H %M")
        agent.save_model(
            f"./backend/lstm_nn/{current_time}.h5",
            f"./backend/scalers/{current_time}.save",
            current_time,
            tickers,
            start_date,
            end_date,
            initial_balance,
            epochs,
            batch_size,
            interval,
            f"./backend/ticker_mappings/{current_time}.json"
        )
        
        # Plot training history
        plot_training_history(current_time)

        # Backtest all tickers at once
        results = backtest_multiple_tickers(agent, tickers, initial_balance=initial_balance)

        # Iterate over results per ticker
        for ticker, (portfolio_values, trades) in results.items():
            print(f"\n===== Backtest Results for {ticker} =====")
            
            # Portfolio timeline plot
            # plot_portfolio_timeline(portfolio_values, title=f"{ticker} Portfolio Timeline")
 
            # Plot trade signals on stock price
            plot_trade_signals(trades, ticker, start_date, end_date)

            # Plot portfolio vs stock price
            plot_backtest_results(portfolio_values, ticker, start_date, end_date)

            # Print summary of trades
            print_trade_summary(trades, portfolio_values, initial_balance)

    except Exception as e:
        print(f"\nError: {str(e)}")
