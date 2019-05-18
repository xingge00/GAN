from matplotlib import pyplot as plt
import pickle

with open('train_samples.pkl', 'rb') as f:
    samples = pickle.load(f)

# 查看各训练轮次生成的图片
epoch_idx = [10, 30, 60, 90, 120, 150, 180, 210, 240, 290]
show_imgs = []
for i in epoch_idx:
    show_imgs.append(samples[i][1])

rows, cols = 10, 25
fig, axes = plt.subplots(figsize=(30, 12), nrows=rows, ncols=cols, sharex=True, sharey=True)

idx = range(0, 300, int(300/rows))
for sample, ax_row in zip(show_imgs, axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
plt.show()
