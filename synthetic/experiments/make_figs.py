import matplotlib.pyplot as plt
import numpy as np

def plot_combined_results(fusion_data, ensemble_data, save_dir, category_dirs):
    # fusion_data and ensemble_data are dictionaries containing train and valid losses and accuracies

    # Create a figure and a set of subplots for accuracy
    fig_acc, axs_acc = plt.subplots(2, 5, figsize=(25, 10))
    # Create a figure and a set of subplots for loss
    fig_loss, axs_loss = plt.subplots(2, 5, figsize=(25, 10))

    # Iterate over each category and its corresponding subplot
    for i, category in enumerate(category_dirs):
        row = i // 5
        col = i % 5

        # Plot for accuracy
        ax_acc = axs_acc[row, col]
        plot_category_data(ax_acc, category, fusion_data, ensemble_data, 'acc', 'Accuracy')

        # Plot for loss
        ax_loss = axs_loss[row, col]
        plot_category_data(ax_loss, category, fusion_data, ensemble_data, 'loss', 'Loss')

    # Adjust layout and save the figures
    fig_acc.tight_layout()
    fig_acc.savefig(f'{save_dir}/combined_accuracy_plot.png')
    plt.close(fig_acc)

    fig_loss.tight_layout()
    fig_loss.savefig(f'{save_dir}/combined_loss_plot.png')
    plt.close(fig_loss)

def plot_category_data(ax, category, fusion_data, ensemble_data, data_type, ylabel):

    # Helper function to plot data for a single category
    epochs = range(len(fusion_data[f'train_{data_type}'][category][0]))  # Assuming all epochs are of the same length

    # Plotting for Fusion
    plot_data(ax, epochs, fusion_data, category, data_type, 'Fusion')

    # Plotting for Ensemble
    plot_data(ax, epochs, ensemble_data, category, data_type, 'Ensemble')

    ax.set_title(f'{category} - {ylabel}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

def plot_data(ax, epochs, data, category, data_type, label_prefix):
    # Helper function to plot train and valid data
    train_mean = np.mean(data[f'train_{data_type}'][category], axis=0)
    train_se = np.std(data[f'train_{data_type}'][category], axis=0) / np.sqrt(len(data[f'train_{data_type}'][category]))
    valid_mean = np.mean(data[f'valid_{data_type}'][category], axis=0)
    valid_se = np.std(data[f'valid_{data_type}'][category], axis=0) / np.sqrt(len(data[f'valid_{data_type}'][category]))

    ax.plot(epochs, train_mean, label=f'{label_prefix} Train {data_type.capitalize()}')
    ax.fill_between(epochs, train_mean - train_se, train_mean + train_se, alpha=0.2)
    ax.plot(epochs, valid_mean, label=f'{label_prefix} Validation {data_type.capitalize()}')
    ax.fill_between(epochs, valid_mean - valid_se, valid_mean + valid_se, alpha=0.2)

import pickle

# Load the dictionary
with open('figs/ensemble_data.pickle', 'rb') as handle:
    ensemble_data = pickle.load(handle)

first = 20

for key in ensemble_data.keys():
    for i in ensemble_data[key].keys():
        ensemble_data[key][i] = ensemble_data[key][i][:, first:]

ensemble_train_accuracies = ensemble_data['ensemble_train_accuracies']
ensemble_train_losses = ensemble_data['ensemble_train_losses']
ensemble_valid_accuracies = ensemble_data['ensemble_valid_accuracies']
ensemble_valid_losses = ensemble_data['ensemble_valid_losses']

with open('figs/fusion_data.pickle', 'rb') as handle:
    fusion_data = pickle.load(handle)

for key in fusion_data.keys():
    for i in fusion_data[key].keys():
        fusion_data[key][i] = fusion_data[key][i][:, first:]

fusion_train_accuracies = fusion_data['fusion_train_accuracies']
fusion_train_losses = fusion_data['fusion_train_losses']
fusion_valid_accuracies = fusion_data['fusion_valid_accuracies']
fusion_valid_losses = fusion_data['fusion_valid_losses']

fusion_data = {
    'train_loss': fusion_train_losses,
    'train_acc': fusion_train_accuracies,
    'valid_loss': fusion_valid_losses,
    'valid_acc': fusion_valid_accuracies
}

ensemble_data = {
    'train_loss': ensemble_train_losses,
    'train_acc': ensemble_train_accuracies,
    'valid_loss': ensemble_valid_losses,
    'valid_acc': ensemble_valid_accuracies
}

dirs = ["redundancy", "uniqueness0", "uniqueness1", "synergy", "mix1", "mix2", "mix3", "mix4", "mix5", "mix6"]
figdir = "figs"

plot_combined_results(fusion_data, ensemble_data, figdir, dirs)
