import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import save_img
from tqdm import tqdm

import csv
import os
import signal


tf.config.run_functions_eagerly(True)


class GAN:
    CLASSES = 43
    LATENT_DIM = 100

    def __init__(self):
        self.dataset = None

        self.gen = None
        self.gen_opt = None
        self.gen_loss = None
        self.disc = None
        self.disc_opt = None
        self.disc_loss = None

        self.cp = None

        self.disc_losses1 = None
        self.disc_losses2 = None
        self.gen_losses = None

        self.run = True
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def set_dataset(self, dataset, buffer_size=10000, batch_size=64):
        self.dataset = dataset.shuffle(buffer_size).batch(batch_size)

    def set_models(self, gen, disc):
        self.gen = gen
        self.gen_opt = tf.keras.optimizers.Adam(1e-4)
        self.gen_loss = tf.keras.losses.BinaryCrossentropy()
        self.disc = disc
        self.disc_opt = tf.keras.optimizers.Adam(1e-4)
        self.disc_loss = tf.keras.losses.BinaryCrossentropy()

        self.cp = tf.train.Checkpoint(gen=self.gen,
                                      gen_opt=self.gen_opt,
                                      disc=self.disc,
                                      disc_opt=self.disc_opt)

    def load_models(self, cp_path: str):
        self.cp.restore(cp_path)

    @tf.function
    def train_step(self, images, target):
        noise = tf.random.normal([target.shape[0], self.LATENT_DIM])
        with tf.GradientTape() as disc_tape1:
            real_output = self.disc([images, target], training=True)
            real_targets = tf.ones_like(real_output)
            disc_loss1 = self.disc_loss(real_targets, real_output)

        gradients_of_disc1 = disc_tape1.gradient(disc_loss1, self.disc.trainable_variables)
        self.disc_opt.apply_gradients(zip(gradients_of_disc1, self.disc.trainable_variables))

        with tf.GradientTape() as disc_tape2:
            generated_images = self.gen([noise, target], training=True)
            fake_output = self.disc([generated_images, target], training=True)
            fake_targets = tf.zeros_like(fake_output)
            disc_loss2 = self.disc_loss(fake_targets, fake_output)

        gradients_of_disc2 = disc_tape2.gradient(disc_loss2, self.disc.trainable_variables)
        self.disc_opt.apply_gradients(zip(gradients_of_disc2, self.disc.trainable_variables))

        with tf.GradientTape() as gen_tape:
            generated_images = self.gen([noise, target], training=True)
            fake_output = self.disc([generated_images, target], training=True)
            real_targets = tf.ones_like(fake_output)
            gen_loss = self.gen_loss(real_targets, fake_output)

        gradients_of_gen = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        self.gen_opt.apply_gradients(zip(gradients_of_gen, self.gen.trainable_variables))

        return disc_loss1.numpy(), disc_loss2.numpy(), gen_loss.numpy()

    def train(self, cp_dir: str, out_dir: str, epochs=300):
        self.disc_losses1 = []
        self.disc_losses2 = []
        self.gen_losses = []

        for epoch in range(epochs):
            epoch_desc = f"Epoch {epoch + 1}/{epochs}"
            epoch_iter = tqdm(iter(self.dataset), desc=epoch_desc, total=tf.data.experimental.cardinality(self.dataset).numpy())

            disc_losses1 = []
            disc_losses2 = []
            gen_losses = []

            for image_batch, label in self.dataset:
                if not self.run:
                    return

                disc_loss1, disc_loss2, gen_loss = self.train_step(image_batch, label)
                epoch_iter.set_postfix({"disc_loss1": disc_loss1, "disc_loss2": disc_loss2, "gen_loss": gen_loss})
                epoch_iter.update(1)

                disc_losses1.append(disc_loss1)
                disc_losses2.append(disc_loss2)
                gen_losses.append(gen_loss)

                self.disc_losses1.append(disc_loss1)
                self.disc_losses2.append(disc_loss2)
                self.gen_losses.append(gen_loss)

            avg_disc_loss1 = np.mean(disc_losses1)
            avg_disc_loss2 = np.mean(disc_losses2)
            avg_gen_loss = np.mean(gen_losses)

            epoch_iter.set_postfix({"disc_loss1": avg_disc_loss1, "disc_loss2": avg_disc_loss2, "gen_loss": avg_gen_loss})

            if (epoch + 1) % 20 == 0:
                self.cp.save(cp_dir)

            self._save_imgs(epoch + 1, out_dir)

    def _save_imgs(self, epoch: int, out_dir: str):
        noise = np.random.normal(0, 1, (25, self.LATENT_DIM))

        sampled_labels = np.random.randint(0, self.CLASSES, 25)
        gen_imgs = self.gen.predict([noise, sampled_labels], verbose=0)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(5, 5, figsize=(10, 10))
        fig.suptitle(f"Epoch {epoch}")
        cnt = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig(f"{out_dir}/{epoch}.png")
        plt.close()

    def _stop(self, *args):
        self.run = False

    def import_losses(self, file_name: str):
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['gen_loss', 'disc_loss1', 'disc_loss2'])
            for i in range(len(self.gen_losses)):
                writer.writerow([self.gen_losses[i], self.disc_losses1[i], self.disc_losses2[i]])

    def generate_all(self, count: int, out_dir: str):
        for cl in range(self.CLASSES):
            noise = np.random.normal(0, 1, (count, self.LATENT_DIM))

            sampled_labels = np.asarray([cl] * count)
            gen_imgs = self.gen.predict([noise, sampled_labels], verbose=0)
            gen_imgs = 0.5 * gen_imgs + 0.5

            for i in range(len(gen_imgs)):
                im = gen_imgs[i, :, :, :]
                path = os.path.join(out_dir, f'{cl}')
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, f'{i}.png')
                save_img(path, im)

