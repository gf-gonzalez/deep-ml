import numpy as np

def pos_encoding(position: int, d_model: int):
	if position == 0 or d_model <= 0:
		return -1
	pos_encoding = np.zeros((position, d_model))
	for i in range(d_model//2):
		pos_encoding[:, 2*i] = np.sin(np.arange(position)/(10000**((2*i)/d_model)))
		pos_encoding[:, 2*i + 1] = np.cos(np.arange(position)/(10000**((2*i)/d_model)))
	pos_encoding = np.float16(pos_encoding)
	return pos_encoding