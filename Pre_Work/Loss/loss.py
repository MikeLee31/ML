import torch
import torch.nn as nn
import math

torch.manual_seed(20)

def validate_loss(output, target):
    val = 0
    for li_x, li_y in zip(output, target):
        for i, xy in enumerate(zip(li_x, li_y)):
            x, y = xy
            loss_val = y * (math.log(y, math.e) - x)
            val += loss_val
    return val / output.nelement()




loss = nn.KLDivLoss()
input = torch.Tensor([[-2, -6, -8], [-7, -1, -2], [-1, -9, -2.3], [-1.9, -2.8, -5.4]])
target = torch.Tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.5, 0.2, 0.3], [0.4, 0.3, 0.3]])
output = loss(input, target)
print("default loss:", output)

output = validate_loss(input, target)
print("validate loss:", output)

loss = nn.KLDivLoss(reduction="batchmean")
output = loss(input, target)
print("batchmean loss:", output)

loss = nn.KLDivLoss(reduction="mean")
output = loss(input, target)
print("mean loss:", output)

loss = nn.KLDivLoss(reduction="none")
output = loss(input, target)
print("none loss:", output)