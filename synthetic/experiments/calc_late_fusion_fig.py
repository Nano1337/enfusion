import os
import re
import numpy as np 
import matplotlib.pyplot as plt

def plot_results(train_losses, train_accuracies, valid_losses, valid_accuracies, save_dir, category_dirs):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 5, figsize=(25, 10))

    # Iterate over each category and its corresponding subplot
    for i, category in enumerate(category_dirs):
        row = i // 5
        col = i % 5

        # Calculating mean and standard error for train and validation accuracies
        train_acc_mean = np.mean(train_accuracies[category], axis=0)
        train_acc_se = np.std(train_accuracies[category], axis=0) / np.sqrt(len(train_accuracies[category]))
        valid_acc_mean = np.mean(valid_accuracies[category], axis=0)
        valid_acc_se = np.std(valid_accuracies[category], axis=0) / np.sqrt(len(valid_accuracies[category]))

        # Plotting train accuracy and validation accuracy with shaded standard error
        ax = axs[row, col]
        epochs = range(len(train_acc_mean))
        ax.plot(epochs, train_acc_mean, label='Train Accuracy')
        ax.fill_between(epochs, train_acc_mean - train_acc_se, train_acc_mean + train_acc_se, alpha=0.2)
        ax.plot(epochs, valid_acc_mean, label='Validation Accuracy')
        ax.fill_between(epochs, valid_acc_mean - valid_acc_se, valid_acc_mean + valid_acc_se, alpha=0.2)
        ax.set_title(category + ' - Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the entire figure
    plt.savefig(f'{save_dir}/fusion_accuracy_plot.png')
    plt.close(fig)

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 5, figsize=(25, 10))

    # Iterate over each category and its corresponding subplot
    for i, category in enumerate(category_dirs):
        row = i // 5
        col = i % 5

        # Calculating mean and standard error for train and validation losses
        train_loss_mean = np.mean(train_losses[category], axis=0)
        train_loss_se = np.std(train_losses[category], axis=0) / np.sqrt(len(train_losses[category]))
        valid_loss_mean = np.mean(valid_losses[category], axis=0)
        valid_loss_se = np.std(valid_losses[category], axis=0) / np.sqrt(len(valid_losses[category]))

        # Plotting train loss and validation loss with shaded standard error
        ax = axs[row, col]
        epochs = range(len(train_loss_mean))
        ax.plot(epochs, train_loss_mean, label='Train Loss')
        ax.fill_between(epochs, train_loss_mean - train_loss_se, train_loss_mean + train_loss_se, alpha=0.2)
        ax.plot(epochs, valid_loss_mean, label='Validation Loss')
        ax.fill_between(epochs, valid_loss_mean - valid_loss_se, valid_loss_mean + valid_loss_se, alpha=0.2)
        ax.set_title(category + ' - Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the entire figure
    plt.savefig(f'{save_dir}/fusion_loss_plot.png')
    plt.close(fig)




# Function to extract the integer from the filename
def extract_integer(filename):
    parts = filename.split('_')  # Split the filename at underscores
    if parts:
        last_part = parts[-1]  # Get the last part of the split
        number_part = last_part.split('.')[0]  # Split at the period and take the first part
        return int(number_part)  # Convert to integer
    return 0  # Return 0 if the filename doesn't match the expected format


dirs = ["redundancy", "uniqueness0", "uniqueness1", "synergy", "mix1", "mix2", "mix3", "mix4", "mix5", "mix6"]
figdir = "figs"

# Regex pattern to match files with the specified format
# It assumes 'name' is a sequence of characters (excluding underscores) and 'int' is an integer
pattern = r'^[^_]+_additive_\d+\.txt$'

# Regex patterns to match the required data
train_pattern = r'Epoch \d+ train loss: ([\d\.]+) acc: ([\d\.]+)'
valid_pattern = r'Epoch \d+ valid loss: ([\d\.]+) acc: ([\d\.]+)'

dict_train_losses = {}
dict_train_accuracies = {}
dict_valid_losses = {}
dict_valid_accuracies = {}

# go through each type of dataset
for dir in dirs: 
    files = os.listdir(f"{dir}/")
    files = [os.path.join(dir,file) for file in files if re.match(pattern, file)]   
    files = sorted(files, key=extract_integer)

    cat_train_losses = []
    cat_train_accuracies = []
    cat_valid_losses = []
    cat_valid_accuracies = []

    # go through each seed in a particular dataset
    for f in files[:3]: 

        # Initialize lists to store the extracted data
        train_losses = []
        train_accuracies = []
        valid_losses = []
        valid_accuracies = []

        print(f'Processing {f}...')

        # Read the file and process each line
        with open(f, 'r') as file:
            for line in file:
                # Check for train data
                train_match = re.search(train_pattern, line)
                if train_match:
                    train_losses.append(float(train_match.group(1)))
                    train_accuracies.append(float(train_match.group(2)))

                # Check for validation data
                valid_match = re.search(valid_pattern, line)
                if valid_match:
                    valid_losses.append(float(valid_match.group(1)))
                    valid_accuracies.append(float(valid_match.group(2)))
    
        # Append the extracted data to the list for the current category
        cat_train_losses.append(train_losses)
        cat_train_accuracies.append(train_accuracies)
        cat_valid_losses.append(valid_losses)
        cat_valid_accuracies.append(valid_accuracies)

    dict_train_accuracies.update({dir: np.array(cat_train_accuracies)})
    dict_train_losses.update({dir: np.array(cat_train_losses)})
    dict_valid_accuracies.update({dir: np.array(cat_valid_accuracies)})
    dict_valid_losses.update({dir: np.array(cat_valid_losses)})

# pickle dictionaries
import pickle

# Save the dictionary
with open('figs/fusion_data.pickle', 'wb') as handle:
    pickle.dump({
        'fusion_train_accuracies': dict_train_accuracies,
        'fusion_train_losses': dict_train_losses,
        'fusion_valid_accuracies': dict_valid_accuracies,
        'fusion_valid_losses': dict_valid_losses
    }, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Plot the results
# plot_results(dict_train_losses, dict_train_accuracies, dict_valid_losses, dict_valid_accuracies, figdir, dirs)