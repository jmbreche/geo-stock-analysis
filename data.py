import glob
import os
import re

import numpy as np
import pandas as pd

from alive_progress import alive_bar
from sec_edgar_downloader import Downloader


df = pd.DataFrame(columns=(["Ticker", "Date", "State", *range(10)]))

paths = glob.glob("data/stocks/*.parquet")

dl = Downloader("Jacob Brecheisen", "brecheisen.jacob@gmail.com", "data/edgar")

biweekly = np.array([1, 1, 1, 1, 3, 1, 1, 1, 1])

stocks = []

with alive_bar(len(paths)) as bar:
    for path in paths:
        stock = pd.read_parquet(path, columns=["Close"])

        if len(stock) < 10:
            continue

        days = pd.Series(stock.index).diff().dt.days.values

        days_window = np.lib.stride_tricks.sliding_window_view(days, 10)
        close_window = np.lib.stride_tricks.sliding_window_view(stock["Close"].values, 10)

        mask = np.all(days_window[:, 1:] == biweekly, axis=1)

        I = stock.index[np.where(mask)[0]]

        ticker = path.replace("\\", "/").split("/")[-1].split(".")[0]

        try:
            dl.get("10-K", ticker, limit=1)

            path = f"data/edgar/sec-edgar-filings/{ticker}/10-K"

            _, dirs, _ = next(os.walk(path))

            path = f"{path}/{dirs[0]}/full-submission.txt"

            with open(path, "r") as file:
                content = file.read()

            match = re.search(r'BUSINESS ADDRESS:.*?STATE:\s*([^\n\r]*)', content, re.DOTALL)
                
            state = match.group(1).strip()
        except Exception:
            state = "NA"

        stock = pd.DataFrame({"Ticker": ticker, "Date": I, "State": state}, columns=(["Ticker", "Date", "State", *range(10)]))

        stock.loc[:, 0:9] = close_window[mask]

        stocks.append(stock)

        bar()

df = pd.concat(stocks)

df.loc[:, 0:9] = df.loc[:, 0:9].divide(df[0], axis=0)

df.set_index(["Ticker", "Date"], inplace=True)

df.columns = df.columns.map(str)

df.to_parquet("data/full.parquet")
