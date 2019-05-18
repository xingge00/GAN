import tensorflow as tf
import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("/home/kesci/input/mnist")


def get_inputs(real_size, noise_size):
    real_img = tf.placeholder(tf.float32, [None, real_size])
    noise_img = tf.placeholder(tf.float32, [None, noise_size])
    return real_img, noise_img


# 生成器
def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    with tf.variable_scope("generator", reuse=reuse):
        # hidden1 layer
        hidden1 = tf.layers.dense(noise_img, n_units)
        # leaky ReLU
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        # hidden2 layer
        hidden1 = tf.layers.dense(hidden1, n_units)
        # leaky ReLU
        hidden1 = tf.maximum(alpha * hidden1, hidden1)

        # dropout
        hidden1 = tf.layers.dropout(hidden1, rate=0.3)

        # logits
        logits = tf.layers.dense(hidden1, out_dim)
        outputs = tf.tanh(logits)
        return logits, outputs


# 判别器
def get_discriminator(img, n_units, reuse=False, alpha=0.01):
    with tf.variable_scope("discriminator", reuse=reuse):
        # hidden1 layer
        hidden1 = tf.layers.dense(img, n_units)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        # hidden2 layer
        hidden1 = tf.layers.dense(hidden1, n_units)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)

        logits = tf.layers.dense(hidden1, 1)
        outputs = tf.sigmoid(logits)

        return logits, outputs


img_size = mnist.train.images[0].shape[0]
noise_size = 100
g_units = 128
d_units = 128
learning_rate = 0.001
alpha = 0.01


# 构建网络
tf.reset_default_graph()
real_img, noise_img = get_inputs(img_size, noise_size)
# generator
g_logits, g_outputs = get_generator(noise_img, g_units, img_size)
# discriminator
d_logits_real, d_outputs_real = get_discriminator(real_img, d_units)
d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, d_units, reuse=True)


# discriminator的loss
# 识别真实图片
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real)))
# 识别生成图片
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
# 总体loss
d_loss = tf.add(d_loss_real, d_loss_fake)
# generator的loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))


# 优化器
train_vars = tf.trainable_variables()
# generator
g_vars = [var for var in train_vars if var.name.startswith('generator')]
# discriminator
d_vars = [var for var in train_vars if var.name.startswith('discriminator')]
# optimizer
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

# 开始训练
batch_size = 64
epochs = 300
n_sample = 25

# 存储测试样本
samples = []
# 存储loss
losses = []
saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch = mnist.train.next_batch(batch_size)

            batch_images = batch[0].reshape((batch_size, 784))
            # 对图像像素进行scale，这是因为tanh输出的结果介于（-1，1），real和real图片共享discriminator的参数
            batch_images = batch_images * 2 - 1

            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

            # Run optimizer
            _ = sess.run(d_train_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})
            _ = sess.run(g_train_opt, feed_dict={noise_img: batch_noise})

        # 每一轮结束计算loss
        train_loss_d = sess.run(d_loss, feed_dict={real_img: batch_images, noise_img: batch_noise})
        train_loss_d_real = sess.run(d_loss_real, feed_dict={real_img: batch_images, noise_img: batch_noise})
        train_loss_d_fake = sess.run(d_loss_fake, feed_dict={real_img: batch_images, noise_img: batch_noise})
        train_loss_g = sess.run(g_loss, feed_dict={real_img: batch_images, noise_img: batch_noise})

        print("Epoch {}/{}...".format(e + 1, epochs),
              "判别器损失:{:.4f}(判别真实的：{:.4f}+判别生成的：{:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
              "生成器损失：{:.4f}".format(train_loss_g))
        losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))

        # 保存样本
        sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
        gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),
                               feed_dict={noise_img: sample_noise})
        samples.append(gen_samples)
        # 保存模型
        saver.save(sess, './checkpoints/generator.ckpt')

with open('train_samples.pkl', 'wb') as f:
    pickle.dump(samples, f)
# 保存训练时的loss的变化到本地
with open('losses.txt', 'w') as f:
    f.write(str(losses))
print('end...')
