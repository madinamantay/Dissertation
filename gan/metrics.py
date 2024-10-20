import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metrics(file_name: str, output_file: str):
    data = pd.read_csv(file_name)

    x = data.index
    y1 = data['gen_loss']
    y1_red = y1.values.reshape(-1, 810)
    y1_red = np.mean(y1_red, axis=1)
    x_red = np.arange(0, len(y1_red))
    y2 = data['disc_loss1']
    y2_red = y2.values.reshape(-1, 810)
    y2_red = np.mean(y2_red, axis=1)
    y3 = data['disc_loss2']
    y3_red = y3.values.reshape(-1, 810)
    y3_red = np.mean(y3_red, axis=1)

    plt.figure(figsize=(30, 10))

    plt.plot(x_red, y1_red, label='gen_loss')
    plt.plot(x_red, y2_red, label='disc_loss1')
    plt.plot(x_red, y3_red, label='disc_loss2')

    plt.xlabel('Time')
    plt.ylabel('Losses')
    plt.title('Plot of losses')
    plt.legend()

    plt.savefig(output_file)
