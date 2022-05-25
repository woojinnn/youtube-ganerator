import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import tensorflow as tf

from keras import initializers
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, Input, Reshape, ZeroPadding2D, Conv2DTranspose)
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam


# dictionary for mapping category_id and category_name
CATEGORY_DICT = {
    1: "Film&Animation",
    2: "Autos&Vehicles",
    10: "Music",
    15: "Pets&Animals",
    17: "Sports",
    19: "Travel&Events",
    20: "Gaming",
    22: "People&Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News&Politics",
    26: "Howto&Style",
    27: "Education",
    28: "Science&Technology",
    29: "Nonprofits&Activism"
}


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 90
        self.img_cols = 120
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self._build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['binary_accuracy']
        )

        # Build the generator
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.generator = self._build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(inputs=z, outputs=valid)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics='binary_accuracy'
        )

    def _build_generator(self):

        generator = Sequential()
        init = initializers.RandomNormal(stddev=1)

        # FC layer: 5*5*256
        generator.add(Dense(5*5*256, input_shape=(self.latent_dim,),
                            kernel_initializer=init))
        generator.add(Reshape((5, 5, 256)))
        generator.add(BatchNormalization())
        generator.add(ReLU())

        # Conv 1: 15x20x128
        generator.add(Conv2DTranspose(
            128, kernel_size=5, strides=(3, 4), padding='same'))
        generator.add(BatchNormalization())
        generator.add(ReLU())
        assert generator.output_shape == (None, 15, 20, 128)

        # Conv 2: 45*60*64
        generator.add(Conv2DTranspose(
            64, kernel_size=5, strides=3, padding='same'))
        generator.add(BatchNormalization())
        generator.add(ReLU())
        assert generator.output_shape == (None, 45, 60, 64)

        # Conv 3: 90x120x3
        generator.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding='same',
                                      activation='tanh'))
        assert generator.output_shape == (None, 90, 120, 3)

        generator.summary()
        return generator

    def _build_discriminator(self):

        discriminator = Sequential()

        discriminator.add(Conv2D(32, kernel_size=3, strides=2,
                                 input_shape=self.img_shape, padding="same"))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        discriminator.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(
            Conv2D(128, kernel_size=3, strides=2, padding="same"))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(
            Conv2D(256, kernel_size=3, strides=1, padding="same"))
        discriminator.add(BatchNormalization(momentum=0.8))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.25))
        discriminator.add(Flatten())
        discriminator.add(Dense(1, activation='sigmoid'))

        discriminator.summary()

        img = Input(shape=self.img_shape)
        validity = discriminator(img)

        return Model(img, validity)

    # loading data
    def _load_data(self, cat_id):
        images = []
        for img in os.listdir(f'data/data_kr/{cat_id}'):
            images.append(plt.imread(
                f'data/data_kr/{cat_id}/{img}', "jpg"))
        return np.asarray(images)

    # Check whether Directory exists
    def _check_dir(self, path):
        # Check whether the specified path exists or not
        dir_exist = os.path.exists(path)

        if not dir_exist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print(f'Directory {path} is created!')

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        return total_loss

    def train(self, cat_id, epochs, batch_size=128, save_interval=50):
        # Check whether Directory exists
        self._check_dir(f'model/{cat_id}')
        self._check_dir(f'images/{cat_id}')

        # Load the dataset
        X_train = self._load_data(cat_id)
        # Rescale -1 to 1
        X_train = Normalize(0, 255)(X_train)

        # Adversarial ground truths
        fake = np.ones((batch_size, 1))
        valid = np.zeros((batch_size, 1))

        for epoch in range(epochs+1):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Train Discriminator weights
            self.discriminator.trainable = True

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 100, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train Generator weights
            self.discriminator.trainable = False

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print(
                f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%%] [G loss: {g_loss}]")

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self._save_imgs(cat_id, epoch)
                self._save_model(cat_id)

    def _save_imgs(self, cat_id, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)    # (25, 90, 120, 3)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(
                    ((gen_imgs[cnt] + 1) * 127).astype(np.uint8))
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f'images/{cat_id}/yt_{epoch}.png')
        plt.close()

    def _save_model(self, cat_id):
        # Save generator
        model_json = self.generator.to_json()
        with open(f'model/{cat_id}/generator.json', "w") as json_file:
            json_file.write(model_json)
        self.generator.save_weights(f'model/{cat_id}/generator.h5')

        # Save discriminator
        model_json = self.discriminator.to_json()
        with open(f'model/{cat_id}/discriminator.json', "w") as json_file:
            json_file.write(model_json)
        self.generator.save_weights(f'model/{cat_id}/discriminator.h5')

        # Save combined
        model_json = self.combined.to_json()
        with open(f'model/{cat_id}/combined.json', "w") as json_file:
            json_file.write(model_json)
        self.generator.save_weights(f'model/{cat_id}/combined.h5')


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(cat_id="Pets&Animals", epochs=10000,
                batch_size=128, save_interval=200)
    # for category in CATEGORY_DICT.values():
    #     dcgan = DCGAN()
    #     dcgan.train(cat_id=category, epochs=1000,
    #                 batch_size=128, save_interval=200)
