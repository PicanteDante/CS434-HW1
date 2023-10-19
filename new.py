import numpy as np


traindata = np.genfromtxt('train.csv', delimiter=',')[:, 1:]
train_X = traindata[1:, :-1]
train_y = traindata[1:, -1]
train_y = train_y[:,np.newaxis]

data = traindata = np.genfromtxt('train.csv', delimiter=',', names=True, dtype=None)

print((data[1]))
print(type(data[0]))
print(type(data[0]))
print(type(data[0]))
print(type(data[0]))
print(type(data[0]))
print(type(data[0]))
print(data.dtype.names)


#incomelike = sum()
for i in range(0, len(data.dtype.names)):
	column_sum = np.sum(data[:,i])
	print("Sum of values in column ", column_index, " is: ", column_sum)
