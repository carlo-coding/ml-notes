import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from micrograd_from_scratch import MLP

num_points = 100


points = np.linspace(1,  3 * np.pi, num_points)

spiral = np.column_stack([points*np.cos(points), points*np.sin(points)])

X = np.vstack([spiral, -spiral])

X += np.random.uniform(-1, 1, X.shape)

Y = np.concat([np.zeros(num_points), np.ones(num_points)])

print("Data shape")
print("spiral.shape: ", spiral.shape)
print("X.shape: ", X.shape)
print("Y.shape: ", Y.shape)

n = MLP(2, [16, 16, 8, 1])

imgx = np.linspace(-11, 11, 200)
imgy = np.linspace(-11, 11, 200)
img_inf = np.zeros((200, 200))


# TRAINING
epochs = 1000
lr0 = 0.002
decay = 0.97
for epoch in range(epochs):

    lr = lr0 * decay ** (epoch/500)

    # FORWARD
    ypred = [n(x) for x in X]
    # BACKWARD
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(Y, ypred)])
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    # UPDATE
    for p in n.parameters():
        p.data += -lr*p.grad


    if (epoch+1) % 10 == 0:
        print("Epoch %i | Loss %.4f | lr %.8f" % (epoch+1, loss.data, lr))


for i, xi in enumerate(imgx):
    for j, yi in enumerate(imgy):
        img_inf[j, i] = n([xi, yi]).data

plt.imshow(img_inf, extent=[11, -11, 11, -11], alpha=0.5)
plt.scatter(X[: , 0], X[: , 1], c=Y)

plt.savefig(f"{Path(__file__).parent}/classification.png")

# Epoch 50 | Loss 32.6884 | lr 0.00237070
# Epoch 60 | Loss 29.3743 | lr 0.00203580
# Epoch 70 | Loss 24.5787 | lr 0.00174821
# Epoch 80 | Loss 25.4194 | lr 0.00150125
# Epoch 90 | Loss 26.0103 | lr 0.00128918
# Epoch 100 | Loss 19.8131 | lr 0.00110706
# Epoch 110 | Loss 16.5951 | lr 0.00095067
# Epoch 120 | Loss 14.9433 | lr 0.00081637
# Epoch 130 | Loss 14.1084 | lr 0.00070105
# Epoch 140 | Loss 13.5030 | lr 0.00060201
# Epoch 150 | Loss 13.0320 | lr 0.00051697
# Epoch 160 | Loss 12.6591 | lr 0.00044394
# Epoch 170 | Loss 12.3598 | lr 0.00038123
# Epoch 180 | Loss 12.1168 | lr 0.00032737
# Epoch 190 | Loss 11.9176 | lr 0.00028112
# Epoch 200 | Loss 11.7531 | lr 0.00024141
# Epoch 210 | Loss 11.6163 | lr 0.00020731
# Epoch 220 | Loss 11.5020 | lr 0.00017802
# Epoch 230 | Loss 11.4060 | lr 0.00015287
# Epoch 240 | Loss 11.3252 | lr 0.00013128
# Epoch 250 | Loss 11.2569 | lr 0.00011273
# Epoch 260 | Loss 11.1991 | lr 0.00009681
# Epoch 270 | Loss 11.1500 | lr 0.00008313
# Epoch 280 | Loss 11.1083 | lr 0.00007139
# Epoch 290 | Loss 11.0727 | lr 0.00006130
# Epoch 300 | Loss 11.0424 | lr 0.00005264