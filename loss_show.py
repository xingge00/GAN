from matplotlib import pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(20, 7))
with open('losses.txt', 'r') as f:
    losses = eval(f.read())
losses = np.array(losses)
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.plot(losses.T[0], label='判别器总损失')
plt.plot(losses.T[1], label='判别真实损失')
plt.plot(losses.T[2], label='判别生成损失')
plt.plot(losses.T[3], label='生成器损失')
plt.title('对抗生成网络')
ax.set_xlabel('epoch')
plt.legend()
plt.show()
