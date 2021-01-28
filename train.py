import argparse

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
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
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

	count = 100
	input_seq_length = 10
	output_seq_length = 1
	batch_size = input_seq_length + output_seq_length

	data = client.get_historical_klines(opt.symbol, Client.KLINE_INTERVAL_1MINUTE, f"{(count * batch_size) + 1} minutes ago UTC")
	candles = pd.DataFrame(data, columns=["date_open", "open", "high", "low", "close", "volume", "date_close", "volume_asset", "trades", "volume_asset_buy", "volume_asset_sell", "ignore"])

	input_data, output_data = split_df_to_input_output_train(candles, count, batch_size, input_seq_length, output_seq_length)

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

			outputs_pred = model(inputs)[0]
			loss = loss_fn(outputs_pred, outputs)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_loss = loss.item()

			if (i + 1) % (total_step // 10) == 0:
				print("Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}".format(epoch + 1, num_epochs, i + 1, total_step, epoch_loss))

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
