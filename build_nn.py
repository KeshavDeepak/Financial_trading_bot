from datetime import datetime
from lstm_nn_2 import LSTMTradingAgent

if __name__ == "__main__":
    # Initialize the trading agent
    agent = LSTMTradingAgent(look_back=60, train_test_split=0.8)
    
    # Choose data to train on
    # ticker = 'AAPL'
    # start_date = '2015-01-01'
    # end_date = '2023-01-01'
    ticker, start_date, end_date = 'AMD', '2018-01-01', '2023-12-31'
    
    initial_balance = 100
    interval = '1d'
    
    # set model hyperparameters
    epochs = 100
    batch_size = 32
    
    try:
        df = agent.fetch_data(ticker, start_date, end_date, interval)
        
        # Prepare data
        agent.prepare_data()
        
        # Build the model
        agent.build_model()
        
        # Train the model
        agent.train_model(epochs=epochs, batch_size=batch_size)
        
        # Plot training history
        agent.plot_training_history()
        
        # Backtest
        portfolio_value, trades = agent.backtest(initial_balance=initial_balance)
        
        # Plot backtest results
        agent.plot_backtest_results(portfolio_value)
        
        # Plot trade signals on the stock line
        agent.plot_trade_signals(trades)
        
        # Print trade summary
        agent.print_trade_summary(trades, portfolio_value, initial_balance, ticker, start_date, end_date)
        
        # save model
        current_time = datetime.now().strftime("%d %b - %H %M")
        
        agent.save_model("./lstm_nn/" + current_time + ".h5", 
                         "./scalers/" + current_time + ".save",
                         current_time,
                         ticker, start_date, end_date, initial_balance, epochs, batch_size, interval)
        
    except Exception as e:
        print(f"\nError: {str(e)}")