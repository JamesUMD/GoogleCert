import numpy as np
import pandas as pd
import torch

# Basics Week 1
# Create a Tensor
a = torch.tensor([7, 4, 3, 2, 6], dtype=torch.int32)

print(a[0])  # Access an element in the tensor
a.dtype  # data type of tensor
a.type()  # type of tensor
a = a.type(torch.FloatTensor)  # change datatype of tensor
print(a)
a.type()
a.size()  # Size of Tensor
a.ndimension()  # Dimension

a_col = a.view(5, 1)
a_col = a.view(-1,
               1)  # Concert 1-D tensor to 2-D tensor, (-1,1) Assumes that we dont know how many are in list so compiles the full amount

numpy_array = np.array([7, 4, 3, 2, 6, 1, 2, 3])  # creates numpy array
torch_tesnor = torch.from_numpy(numpy_array)  # creats tensor from numpy array
back_to_numpy = torch_tesnor.numpy()  # converts tensor back to numpy array
back_to_numpy

pandas_series = pd.Series([7, 4, 3, 2, 6, 1, 1, 2, 56, 634])  # creats pandas series
pandas_to_torch = torch.from_numpy(pandas_series.values)  # creates tensor from pandas series
torch_to_list = pandas_to_torch.tolist()  # creates list of tesnor
pandas_to_torch[1].item()  # returns item of tensor

# Linear Regression Week 2
