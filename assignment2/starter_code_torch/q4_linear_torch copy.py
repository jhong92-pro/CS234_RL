import torch
import torch.nn as nn

data = torch.randn((5, 5))
print(data)
print(data[torch.arange(data.size(dim=0)), torch.tensor([0,1,1,1,1])])

data = torch.Tensor([1,2,3])
done_mask = torch.tensor([True,False,False])
print(data*torch.logical_not(done_mask))

####
# tensor([[ 1.1328, -0.0392, -0.7076,  0.5610,  0.8010],
#         [-0.0898, -1.4467, -0.7285, -0.1195, -2.1070],
#         [ 0.4547,  1.7739,  0.1664, -1.0242,  0.0474],
#         [ 0.2739,  0.8654, -0.2284,  0.7990,  0.2825],
#         [-0.9779,  1.1102,  1.0023,  0.2493, -2.6588]])

####
# tensor(1.7739)