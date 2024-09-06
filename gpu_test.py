import torch
import time
device = torch.device('cpu')
n = 5000
x = torch.rand((n, n), dtype=torch.float32)
y = torch.rand((n, n), dtype=torch.float32)
x = x.to(device)
y = y.to(device)
for i in range(10):
    start = time.time()
    x * y
    end = time.time()
    print(end - start)