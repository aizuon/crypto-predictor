import argparse

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
from binance.client import Client

from network import *


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--key", type=str, help="API key", required=True)
	parser.add_argument("--secret", type=str, help="API secret", required=True)
	parser.add_argument("--symbol", type=str, help="Symbol to trade", default="BTCUSDT")
	opt = parser.parse_args()

	client = Client(opt.key, opt.secret)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SequencePredictor().to(device)
	model.train()
	loss_fn = nn.MSELoss(reduction="mean")
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

	count = 100
	input_seq_length = 50

	data = client.get_historical_klines(opt.symbol, Client.KLINE_INTERVAL_1HOUR, f"{((count + 1) * (input_seq_length + 1))} hours ago UTC")
	candles = pd.DataFrame(data, columns=["date_open", "open", "high", "low", "close", "volume", "date_close", "volume_asset", "trades", "volume_asset_buy", "volume_asset_sell", "ignore"])
	candles.drop(["date_open", "volume", "date_close", "volume_asset", "trades", "volume_asset_buy", "volume_asset_sell", "ignore"], axis=1, inplace=True)

	sc = MinMaxScaler(feature_range=(0, 1))
	candles = sc.fit_transform(candles)
	joblib.dump(sc, f"./model/crypto_predictor_{opt.symbol}.sc")

	input_data, output_data = split_input_output_train(candles, count, input_seq_length)

	dataset = SequenceDataset(input_data, output_data)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

	total_step = len(dataloader)
	num_epochs = 100
	matplotlib.use("TkAgg")
	plt.ion()
	fig, ax = plt.subplots()
	fig.canvas.set_window_title("Train")
	ax.set(xlabel="Epoch", ylabel="MSE")
	ax.grid()
	loss_hl, = ax.plot([], [])
	fig.tight_layout()
	for epoch in range(num_epochs):
		epoch_loss = 0
		for i, (inputs, outputs) in enumerate(dataloader):
			inputs = inputs.to(device)
			outputs = outputs.to(device)

			outputs_pred = model(inputs)
			loss = loss_fn(outputs_pred, outputs)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_loss = iterative_avg(epoch_loss, loss.item(), i + 1)

			if (i + 1) % (total_step // 10) == 0:
				print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {epoch_loss:.2f}")

		loss_hl.set_xdata(np.append(loss_hl.get_xdata(), epoch + 1))
		loss_hl.set_ydata(np.append(loss_hl.get_ydata(), epoch_loss))
		ax.relim()
		ax.autoscale_view()
		fig.canvas.draw()
		fig.canvas.flush_events()

	torch.save(model.state_dict(), f"./model/crypto_predictor_{opt.symbol}.ckpt")

	plt.savefig(f"./model/crypto_predictor_{opt.symbol}.png")

	print("Finished training model")
	plt.ioff()
	plt.show()
