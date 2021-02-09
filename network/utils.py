import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def split_input_output_train(data, count, input_seq_length):
	input_data = []
	output_data = []

	for i in range(count):
		b = data.iloc[i * (input_seq_length + 1):(i + 1) * (input_seq_length + 1)]

		sc = MinMaxScaler(feature_range=(0, 100))
		b = sc.fit_transform(b)
		
		b_input = b[:input_seq_length]
		b_output = b[input_seq_length:input_seq_length + 1]

		input_data.append(b_input)
		output_data.append(b_output)

	return input_data, output_data

def prepare_for_nn_eval(data):
	input_data = []

	input_data.append([row for row in data])

	return torch.from_numpy(np.asarray(input_data, dtype=np.float32))


def iterative_avg(avg, item, i):
	"""
	To prevent buffer overflow while averaging
	"""

	avg += (item - avg) / i
	return avg
