import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(file_name: str, output_file: str):
    data = pd.read_csv(file_name)

    x = data['epoch']
    y1 = data['acc']
    y2 = data['val_acc']
    y3 = data['loss']
    y4 = data['val_loss']

    plt.figure(figsize=(10, 6))

    plt.plot(x, y1, label='accuracy')
    plt.plot(x, y2, label='validation_accuracy')
    plt.plot(x, y3, label='loss')
    plt.plot(x, y4, label='validation_loss')

    plt.xlabel('Epochs')
    plt.ylabel('Training metrics')
    plt.legend()

    plt.savefig(output_file)
