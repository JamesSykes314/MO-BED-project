import torch
import time
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(device)
n = 10000
x = torch.rand((n, n), dtype=torch.float32)
y = torch.rand((n, n), dtype=torch.float32)
x = x.to(device)
y = y.to(device)
for i in range(10):
    start = time.time()
    x * y
    end = time.time()
    print(end - start)