import yfinance as yf

ticker = 'GC=F'

gold_df = gold_df = yf.download(ticker, start='2022-01-01', end='2024-01-01', interval='1d')

print(gold_df.head())