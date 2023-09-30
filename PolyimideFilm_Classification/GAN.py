from __future__ import print_function, division
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K


class AnoEAN():
    def __init__(self):
        # 输入shape
        self.E_loss = []
        self.D_loss = []
        self.loss_a = []
        self.loss_n = []
        self.loss_z = []
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # 分2类
        self.num_classes = 2
        self.latent_dim = 10
        # adam优化器
        optimizer = Adam(1e-3)
        # 判别模型
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['Loss_Discriminator'], optimizer=optimizer, metrics=None)
        # 生成模型
        self.encoder = self.build_encoder()

        # conbine是生成模型和判别模型的结合
        # 判别模型的trainable为False
        # 用于训练生成模型
        img = Input(shape=self.img_shape)
        z = self.encoder(img)
        self.discriminator.trainable = False
        valid = self.discriminator(z)
        self.combined = Model(img, valid)
        self.combined.compile(loss='Loss_Encoder', optimizer=optimizer)

    def build_encoder(self):

        model = Sequential()
        # 28*28*1 -> 14*14*32
        model.add(Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=3, strides=2, padding='same', name='conv1'))
        model.add(BatchNormalization(momentum=0.8, name='bn_conv1'))
        model.add(Activation("relu"))
        # 14*14*32 -> 7*7*64
        model.add(Conv2D(filters=64, kernel_size=3, strides=2, padding='same', name='conv2'))
        model.add(BatchNormalization(momentum=0.8, name='bn_conv2'))
        model.add(Activation("relu"))
        # 7, 7, 64 -> 4, 4, 128
        model.add(ZeroPadding2D(((0, 1), (0, 1))))
        model.add(Conv2D(filters=128, kernel_size=3, strides=2, padding='same', name='conv3'))
        model.add(BatchNormalization(momentum=0.8, name='bn_conv3'))
        model.add(Activation("relu"))
        # model.add(GlobalAveragePooling2D())
        model.add(Flatten())

        # 全连接
        model.add(Dense(self.latent_dim, name='fc_e', activation = "relu"))
        #

        img = Input(shape=self.img_shape)
        Ex = model(img)
        return Model(img, Ex)

    def build_discriminator(self):

        model = Sequential()
        # 全连接
        model.add(Dense(32, activation="relu", input_dim=self.latent_dim, name='fc_d1'))
        model.add(Dense(16, activation="relu", name='fc_d2'))
        model.add(Dense(2, activation="sigmoid", name='fc_d3'))

        Ex = Input(shape=(self.latent_dim,))
        DEx = model(Ex)

        return Model(Ex, DEx)

    def Loss_D(self, y_true, y_pred):
        if y_true[1] == 0:
            loss = - np.log(1 - y_pred[1])
        else:
            loss = - np.log(y_pred[1])
        return loss

    def Loss_E(self,y_true, y_pred):
        if y_true[0] == 1:
            loss = np.log(1 - y_pred[1])
        else:
            loss = np.log(y_pred[1])
        return loss

    def train(self, epochs, batch_size=128, save_interval=50):
        # 载入数据
        (X_train, Y_train), (_, _) = mnist.load_data()

        # 归一化
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # 建立训练集：5000个1为正常样本，500个其余数字为异常样本
        index = Y_train == 1
        X = X_train[index][0:5000, :, :]
        for i in range(6):
            if i == 1:
                continue
            index = Y_train == i
            X = np.concatenate((X, X_train[index][0:100, :, :]), axis=0)
        X_train = X
        Y_train = np.concatenate((np.ones(5000),np.zeros(500)))
        np.random.seed(120)  # 设置随机种子，让每次结果都一样，方便对照
        np.random.shuffle(X_train)  # 使用shuffle()方法，让输入x_train乱序
        np.random.seed(120)
        np.random.shuffle(Y_train)

        # Adversarial ground truths
        valid = np.ones((batch_size, 2))
        fake = np.zeros((batch_size, 2))

        for epoch in range(epochs):
            loss_a = []
            loss_n = []
            loss_z = []
            loss_d = []
            loss_e = []
            for i in range(len(Y_train)//batch_size):
                imgs = X_train[i*batch_size:(i+1)*batch_size]
                Y = Y_train[i*batch_size:(i+1)*batch_size]
                s1 = sum(Y == 1)
                s2 = sum(Y == 0)

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                E_x = self.encoder.predict(imgs, batch_size=batch_size)

                # 计算loss
                D_E_x = self.combined.predict(imgs, batch_size=batch_size)
                D_z = self.discriminator.predict(noise, batch_size=batch_size)
                D_E_x_n = D_E_x[Y == 1]
                D_E_x_a = D_E_x[Y == 0]
                loss1 = sum(-np.log(1 - D_E_x_a[:, 1])) / sum(Y == 0)
                loss2 = sum(-np.log(1 - D_E_x_n[:, 1])) / sum(Y == 1)
                loss3 = sum(-np.log(D_z[:, 1])) / batch_size
                loss_a.append(loss1)
                loss_n.append(loss2)
                loss_z.append(loss3)

                # 训练D并计算loss
                fake[:, 0] = Y
                weight = np.ones(batch_size)
                weight[Y == 1] = 2 * batch_size / s1
                weight[Y == 0] = 2 * batch_size / s2
                x_train_d = np.concatenate((noise, E_x))
                y_train_d = np.concatenate((valid, fake))
                weight = np.concatenate((2*np.ones(batch_size), weight))
                d_loss = self.discriminator.train_on_batch(x_train_d, y_train_d, sample_weight=weight)
                # d_loss_z = self.discriminator.train_on_batch(noise, valid)
                # d_loss_x = self.discriminator.train_on_batch(E_x, fake, sample_weight=weight)
                # d_loss = np.add(d_loss_x, d_loss_z)
                loss_d.append(d_loss)

                #  训练E并计算loss
                weight = np.ones(batch_size)
                weight[Y == 1] = batch_size / s1
                weight[Y == 0] = batch_size / s2
                e_loss = self.combined.train_on_batch(imgs, fake, sample_weight=weight)
                loss_e.append(e_loss)

            self.D_loss.append(np.mean(loss_d))
            self.E_loss.append(np.mean(loss_e))
            self.loss_a.append(np.mean(loss_a))
            self.loss_n.append(np.mean(loss_n))
            self.loss_z.append(np.mean(loss_z))
            print("%d [D loss: %f] [E loss: %f]" % (epoch, np.mean(loss_d), np.mean(loss_e)))

            # if epoch % save_interval == 0:
            #     self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

    def save_weights(self, filepath):
        self.encoder.save_weights(filepath)

    def load_weights(self, filepath):
        self.encoder.load_weights(filepath)

    def plot_loss(self):
        ax2 = plt.subplot(1,2,1)
        ax1 = plt.subplot(1,2,2)
        epochs = range(len(self.E_loss))

        plt.sca(ax1)
        plt.plot(epochs, self.E_loss, "b", label="E_loss")
        plt.legend()
        plt.plot(epochs, self.D_loss, "r", label="D_loss")
        plt.legend()
        plt.grid(True)

        plt.sca(ax2)
        plt.plot(epochs, self.loss_a, "b", label="loss_a")
        plt.legend()
        plt.plot(epochs, self.loss_n, "r", label="loss_n")
        plt.legend()
        plt.plot(epochs, self.loss_z, "g", label="loss_z")
        plt.legend()
        plt.grid(True)

        plt.show()


if __name__ == '__main__':
    # if not os.path.exists("./images"):
    #     os.makedirs("./images")
    anoean = AnoEAN()
    anoean.train(epochs=50, batch_size=250, save_interval=50)
    anoean.save_weights('.h5/AnoEan_1.0.h5')
    anoean.plot_loss()