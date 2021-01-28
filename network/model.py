import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SequencePredictor(nn.Module):
	def __init__(self):
		super().__init__()

		in_params = 4
		out_params = 4

		hidden_layer_size = 100
		layer_count = 2

		self.__lstm = nn.LSTM(input_size=in_params, hidden_size=hidden_layer_size, num_layers=layer_count, dropout=0.1, batch_first=True)
		self.__fc = nn.Linear(hidden_layer_size, out_params)

	def forward(self, x):
		out, h = self.__lstm(x)

		out = self.__fc(out[:, -1, :])

		return out

class SequenceDataset(Dataset):
	def __init__(self, input_data, output_data):
		super().__init__()

		self.__input_data = input_data
		self.__output_data = output_data

	def __len__(self):
		return len(self.__input_data)

	def __getitem__(self, idx):
		return torch.from_numpy(np.asarray(self.__input_data[idx], dtype=np.float32)), torch.from_numpy(np.asarray(self.__output_data[idx], dtype=np.float32))
