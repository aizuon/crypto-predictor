import numpy as np
import torch


def split_df_to_input_output_train(df, count, batch_size, input_seq_length, output_seq_length):
	input_data = []
	output_data = []

	for i in range(count):
		b = df.iloc[i*batch_size:(i+1)*batch_size]
		b_input = b[:input_seq_length]
		b_output = b[input_seq_length:input_seq_length + output_seq_length]

		input_data.append([[row["open"], row["close"], row["high"], row["low"]] for idx, row in b_input.iterrows()])
		output_data.append([[row["open"], row["close"], row["high"], row["low"]] for idx, row in b_output.iterrows()])

	return input_data, output_data

def prepare_for_nn_eval(df):
	input_data = []

	input_data.append([[row["open"], row["close"], row["high"], row["low"]] for idx, row in df.iterrows()])

	return torch.from_numpy(np.asarray(input_data, dtype=np.float32))
