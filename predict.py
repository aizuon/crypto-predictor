import datetime as dt
import time
import argparse

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import torch
from binance.client import Client

from network import *


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--key", type=str, help="API key", required=True)
	parser.add_argument("--secret", type=str, help="API secret", required=True)
	parser.add_argument("--symbol", type=str, help="Symbol to trade", default="BTCUSDT")
	opt = parser.parse_args()

	client = Client(opt.key, opt.secret)

	input_seq_length = 10
	output_seq_length = 1
	batch_size = input_seq_length + output_seq_length

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SequencePredictor().to(device)
	model.load_state_dict(torch.load(f"./model/crypto_predictor_{opt.symbol}.ckpt"))
	model.eval()

	last_minute = None
	matplotlib.use("TkAgg")
	plt.ion()
	fig = None
	ax = None
	while True:
		now = dt.datetime.now()
		if last_minute != now.minute:
			if last_minute is not None:
				time.sleep(5)

			if fig is not None:
				plt.close(fig)

			data = client.get_historical_klines(opt.symbol, Client.KLINE_INTERVAL_1MINUTE, f"{input_seq_length + 1} minutes ago UTC")[:input_seq_length]
			if len(data) != 10:
				continue

			candles = pd.DataFrame(data, columns=["date_open", "open", "high", "low", "close", "volume", "date_close", "volume_asset", "trades", "volume_asset_buy", "volume_asset_sell", "ignore"])

			with torch.no_grad():
				input_data = prepare_for_nn_eval(candles).to(device)

				output_pred = model(input_data)[0]

				candles = candles.append({"date_open": candles["date_open"].iloc[-1] + 60000, "open": output_pred[0].item(), "close": output_pred[1].item(), "high": output_pred[2].item(), "low": output_pred[3].item()}, ignore_index=True)

			candles["date_open"] = pd.to_datetime(candles["date_open"], unit="ms")
			candles["open"] = candles["open"].astype(float)
			candles["close"] = candles["close"].astype(float)
			candles["high"] = candles["high"].astype(float)
			candles["low"] = candles["low"].astype(float)
			candles.rename(columns={"date_open": "Date", "open": "Open", "close": "Close", "high": "High", "low": "Low"}, inplace=True)
			candles.set_index("Date", inplace=True)

			fig, ax = mpf.plot(candles, type="candle", block=False, returnfig=True)
			fig.show()

			last_minute = now.minute
		else:
			if fig is not None:
				fig.canvas.draw()
				fig.canvas.flush_events()
