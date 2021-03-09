from sklearn.datasets import load_files

def pull_training_data(path):
    train_data = load_files(path)
    return train_data


def pull_test_data(path):
    test_data = load_files(path)
    return test_data

