from datasets import load_dataset

# Load the Open-Orca dataset
open_orca_dataset = load_dataset('Open-Orca/OpenOrca')

# Access the dataset splits (e.g., 'train', 'validation', 'test')
train_data = open_orca_dataset['train']
val_data = open_orca_dataset['validation']
test_data = open_orca_dataset['test']

# Example: Print the first example from the training set
print(train_data[0])