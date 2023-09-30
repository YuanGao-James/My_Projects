from __future__ import print_function, division
import tensorflow as tf
from keras.datasets import mnist
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

class ACGAN():
    def __init__(self):
        # input shape
        self.img_rows = 112
        self.img_cols = 112
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 2
        self.latent_dim = 100
        # adam
        optimizer = Adam(0.0002, 0.5)
        # loss for Discriminator
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Generator
        self.generator = self.build_generator()

        # conbine D and G
        # to train G
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        self.discriminator.trainable = False

        valid, target_label = self.discriminator(img)

        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
                              optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        # FC to 32*7*7
        model.add(Dense(32 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        # reshape
        model.add(Reshape((7, 7, 32)))

        # 7, 7, 64
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        # up sampling
        # 7, 7, 64 -> 14, 14, 128
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        # up sampling
        # 14, 14, 128 -> 28, 28, 128
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        # up sampling
        # 28, 28, 128 -> 56, 56, 128
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        # up sampling
        # 56, 56, 128 -> 112, 112, 128
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # up sampling
        # 112, 112, 128 -> 112, 112, 1
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()
        # 112,112,1 -> 56,56,16
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # 56,56,16 -> 28,28,32
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        # 28,28,32 -> 14,14,32
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        # 14,14,32 -> 7,7,32
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        # 7,7,32 -> 4,4,64
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        # 4,4,64 -> 4,4,128
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(GlobalAveragePooling2D())

        img = Input(shape=self.img_shape)

        features = model(img)

        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size=256, sample_interval=50):

        # (X_train, y_train), (_, _) = mnist.load_data()
        # load dataset
        phase3_dir = 'D:/高源/BU/AICV Lab/BML Generate/Phase3'
        imgs = []
        labels = []
        for patient_name in os.listdir(phase3_dir):
            # if patient_name == '9215390':
            #     break
            for img_name in os.listdir(phase3_dir + '/' + patient_name + '/' + 'v00/meta/images'):
                img_add = phase3_dir + '/' + patient_name + '/' + 'v00/meta/images/' + img_name
                img = cv2.imdecode(np.fromfile(img_add, dtype=np.uint8), 0)
                img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
                imgs.append(img)
                label = 0
                if 'bml_masks' in os.listdir(phase3_dir + '/' + patient_name + '/' + 'v00/meta/'):
                    if img_name[:-4] + '_mask' + '.bmp' in os.listdir(
                            phase3_dir + '/' + patient_name + '/' + 'v00/meta/bml_masks'):
                        label = 1
                labels.append(label)
        X_train = np.array(imgs)
        y_train = np.array(labels)

        # normalization
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # visualization for training process
        plt.ion()  # interactive mode
        plt.figure(1)
        metrics = [[], [], [], [], []]
        for epoch in range(epochs):

            # --------------------- #
            #  train D
            # --------------------- #
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # ---------------------- #
            #   input in norm distribution
            # ---------------------- #
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            sampled_labels = np.random.randint(0, 2, (batch_size, 1))
            gen_imgs = self.generator.predict([noise, sampled_labels])

            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # --------------------- #
            #  train G
            # --------------------- #
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
            epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

            if epoch % sample_interval == 0:
                plt.clf()
                self.sample_images(epoch)

                plt.clf()
                metrics[0].append(epoch)
                metrics[1].append(d_loss[0])
                metrics[2].append(100 * d_loss[3])
                metrics[3].append(100 * d_loss[4])
                metrics[4].append(g_loss[0])

                plt.subplot(1, 2, 1)
                plt.plot(metrics[0], metrics[1], 'r', label='D_loss')
                plt.plot(metrics[0], metrics[4], 'b', label='G_loss')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(metrics[0], metrics[2], 'r', label='Fake_Acc')
                plt.plot(metrics[0], metrics[3], 'b', label='Class_Acc')
                plt.legend()

                plt.pause(0.1)
        plt.show()

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        # sampled_labels = np.arange(0, 2).reshape(-1, 1)
        sampled_labels = np.zeros((2, 5))
        sampled_labels[1, :] = 1
        sampled_labels = sampled_labels.reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].set_title("BML: %d" % sampled_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    if not os.path.exists("./images"):
        os.makedirs("./images")
    acgan = ACGAN()
    acgan.train(epochs=20000, batch_size=16, sample_interval=100)
