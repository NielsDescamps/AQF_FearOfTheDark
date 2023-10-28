import pandas as pd
import yfinance as yf

symbol = "SEDG"
save_dir = "data_10_27_2023"
sedg = yf.Ticker(symbol)

expiry_dates = yf.Ticker(symbol).options

for expiry_date in expiry_dates:
    opt = sedg.option_chain(expiry_date)
    filename_calls = "{}/{}_call_chain_{}.csv".format(save_dir,symbol,expiry_date)
    filename_puts = "{}/{}_put_chain_{}.csv".format(save_dir,symbol,expiry_date)
    opt[0].to_csv(filename_calls)
    opt[1].to_csv(filename_puts)
print(expiry_dates)