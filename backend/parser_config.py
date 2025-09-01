
#* imports
import io
import base64

import matplotlib.pyplot as plt

import yfinance as yf

#* show [ticker] [start] [end]
def show(ticker, start, end):
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
    
    # save the plot so it can be returned to the front-end
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close() # to avoid errors
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")\
    
    return img_base64
    
    # plt.show()

#* normalize the command
#* Convert name of company to appropriate ticker name
def normalize_command(command):
    command_components = command.split()
    
    if len(command_components) < 2: return [command]
    
    command_components[1] = COMPANY_TO_TICKER[command_components[1]]
    
    return command_components
    

#* the model is setup as an user input parser
SETUP_MODEL_PROMPT = {
    'role': "system",
    'content':
        '''
            You are a trading command parser. Your job is to take the user's text and convert it into exactly one of these commands:
            - buy [ticker]
            - sell [ticker]
            - show [ticker] [start] [end]  
                [start] and [end] must strictly be in "yyyy-mm-dd" format
                starting time must be before (chronologically) ending time
            
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

#* unused setup
'''
Examples:
            User: I want to purchase Apple stock -> buy AAPL
            User: Sell Tesla please -> sell TSLA
            User: What's Google doing this week? -> show GOOGL week
            User: Show me Microsoft -> show MSFT


            - You must always output ONLY the ticker symbol (from the list below), never the company name
            - If the user provides a company name, replace it with its corresponding ticker from the mapping list
            - If the company name is not in the list, return "error"
            
Mapping list :
                Apple → AAPL  
                Microsoft → MSFT  
                Google → GOOGL  
                Alphabet → GOOGL  
                Tesla → TSLA  
                Amazon → AMZN  
                Meta → META  
                Facebook → META  
                Nvidia → NVDA  
                Netflix → NFLX  
                Intel → INTC  
                AMD → AMD  
                Qualcomm → QCOM  
                IBM → IBM  
                Oracle → ORCL  
                Adobe → ADBE  
                Salesforce → CRM  
                PayPal → PYPL  
                Uber → UBER  
                Lyft → LYFT  
                Snap → SNAP  
                Twitter → TWTR  
                Coca-Cola → KO  
                Pepsi → PEP  
                McDonald’s → MCD  
                Starbucks → SBUX  
                Walmart → WMT  
                Costco → COST  
                Nike → NKE  
                Disney → DIS  
                Sony → SONY  
                Samsung → SMSN.IL  
                Alibaba → BABA  
                Tencent → TCEHY  
                Berkshire Hathaway → BRK.B  
                JPMorgan → JPM  
                Goldman Sachs → GS  
                Bank of America → BAC  
                Morgan Stanley → MS  
                Citigroup → C  
                Visa → V  
                Mastercard → MA  
                American Express → AXP
'''



