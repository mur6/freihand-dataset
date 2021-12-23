import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

print(torch.__version__)  # 1.1.0

w = 4.5
b = 7.0

x_array = np.array([-1.5, -1.0, -0.1, 0.9, 1.8, 2.2, 3.1])
y_array = w * x_array + np.random.normal(size=x_array.shape[0])
#plt.scatter(x_array, y_array)
#plt.show()

x = torch.tensor(x_array).float()
y = torch.tensor(y_array).float()
#a = torch.arange(-10, 10)
#print(a)

param_w = torch.tensor([1.0], requires_grad=True)
param_b = torch.tensor([0.0], requires_grad=True)

learning_rate = 0.01
optimizer = optim.SGD([param_w, param_b], lr=learning_rate)

epochs = 300
for epoch in range(1, epochs + 1):
    y_p = param_w * x + param_b
    loss = torch.mean((y_p - y)**2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss}, param_w={float(param_w)}, param_b={float(param_b)}")
