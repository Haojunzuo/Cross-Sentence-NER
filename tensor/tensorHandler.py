import torch
#shape为（2，2，3）
# a=torch.tensor([[[1,2,3],[4,5,6]],
#                 [[7,8,9],[10,11,12]]])
# #选择索引0和索引2的tensor
# indices=torch.tensor([1])
# #tensor为a,维度为2，索引为0和2
# b=torch.index_select(a,1,indices)
# print(b)
for i in reversed(range(4)):
    print(i)
