
#* imports
import io
import base64

#* make matplotlib run in a non-interactive manner
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yfinance as yf

from backend.utils import *

#* ------------------------------------------------------ CONSTANTS ---------------------------------------------------

#* the name of the agent being used
MODEL_NAME = '05 Aug - 17 57'

#* the model is setup as an user input parser
SETUP_MODEL_FOR_USER_PROMPT = {
    'role': "system",
    'content':
        '''
            You are a trading command parser. Your job is to take the user's text and convert it into exactly one of these commands:
            - suggest [ticker]
            - show [ticker] [start] [end]  
                [start] and [end] must strictly be in "yyyy-mm-dd" format
                starting time must be before (chronologically) ending time
            - explain [concept]
            - backtest [ticker]
            
            Functionality of each of the above commands 
            - 'suggest' is chosen when the user is asking for guidance on whether to buy/sell a particular ticker's stocks
            - 'show' is chosen when the user wants to see past stock data for a given ticker
            - 'explain' is chosen when the user wants an explanation of a particular concept (will most likely be related to the domain of stocks)
            - 'backtest' is chosen when the user wants to backtest a ticker
            
            If you are not able to convert the user prompt into one of the commands, return:
            - error
            
            RULES:
            - Correct any spelling mistakes in the company's name and return the proper name of the company
            - Your reply must be only the command, nothing else
            - All commands and their respective parameters must be filled in, nothing is to be left out
            
            
        '''
}

COMPANY_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "intel": "INTC",
    "amd": "AMD",
    "qualcomm": "QCOM",
    "ibm": "IBM",
    "oracle": "ORCL",
    "adobe": "ADBE",
    "salesforce": "CRM",
    "paypal": "PYPL",
    "uber": "UBER",
    "lyft": "LYFT",
    "snap": "SNAP",
    "twitter": "TWTR",
    "coca-cola": "KO",
    "pepsi": "PEP",
    "mcdonalds": "MCD",
    "mcdonald’s": "MCD",
    "starbucks": "SBUX",
    "walmart": "WMT",
    "costco": "COST",
    "nike": "NKE",
    "disney": "DIS",
    "sony": "SONY",
    "samsung": "SMSN.IL",
    "alibaba": "BABA",
    "tencent": "TCEHY",
    "berkshire hathaway": "BRK.B",
    "jpmorgan": "JPM",
    "goldman sachs": "GS",
    "bank of america": "BAC",
    "morgan stanley": "MS",
    "citigroup": "C",
    "visa": "V",
    "mastercard": "MA",
    "american express": "AXP",
}

#* ------------------------------------------------------ functions ---------------------------------------------------------

#* show [ticker] [start] [end]
def show(ticker, start, end):
    """
    Fetches and visualizes the closing price of a stock over a given time period.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA').
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.
    """
    
    #* Download historical data
    data = yf.download(ticker, start=start, end=end)

    #* Check if data was returned
    if data.empty:
        print(f"No data found for {ticker} between {start} and {end}.")
        return

    #* Plot the closing price
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.title(f"{ticker} Stock Price from {start} to {end}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    #* save the plot so it can be returned to the front-end
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close() # to avoid errors
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    #* return the plot
    return img_base64

#* should [ticker] be bought in the current timeframe or not
def suggest(ticker):
    """
    Uses the AI agent to get a prediction of if the [ticker] is going up/down/neutral and accordingly give an
    appropriate response
    
    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA').
    """
    
    #* return the answer and a system_prompt explaining how to reform the answer
    return (
        str(predict_next_day(ticker, MODEL_NAME)),
        f'''
        The next input is going to consist of two elements :
        1. EIther 'up', 'down' or 'neutral' -- indicates whether {ticker} stock is going to move up/down/not-at-all
        2. A list of three elements consisting of the confidence levels (in probabilities) of the AI agent predicting the stock to move 
        down/neutral/up respectively
        
        The user has asked us to predict the stocks and the next input is the answer the agent has given. 
        Use the input to craft a human-readable response that can be shown back to the user
        Return just the human-readable response and nothing else.
        
        If the prediction is up -- then suggest to buy shares
        If the prediction is down -- then suggest to sell shares
        In all other cases -- suggest to wait or hold their position and mention that neither a up or a down is being predicted
        
        Do not use the word "AI agent", assume you are the AI agent
        '''
    )

#* normalize the command
#* Convert name of company to appropriate ticker name
def normalize_command(command):
    #* split the command up into its components
    command_components = command.split()
    
    #* if command doesn't include a ticker, do not run the remaining code 
    if command_components[0] == "explain": return command_components
    
    #* normalize the name of the ticker (command_components[1]) to that accepted by yfinance
    #* -- convert the ticker to lower case for easier processing
    command_components[1] = command_components[1].lower()
    
    #* -- if the ticker given in is already in a standard format, leave it as is
    if command_components[1] not in COMPANY_TO_TICKER.values(): 
        #* otherwise convert it to a standard format
        command_components[1] = COMPANY_TO_TICKER[command_components[1].lower()]    
    
    #* return the command components
    return command_components

#* backtest the given ticker
def get_portfolio_plot(ticker):
    #* backtest the ticker and store the portfolio values and trades 
    portfolio_values, trades = backtest(ticker, MODEL_NAME)
    
    #* get the plot of the portfolio values
    plot = portfolio_timeline_plot(portfolio_values)
    
    #* return the plot
    return plot




