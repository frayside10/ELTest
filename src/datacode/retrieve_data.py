from sklearn.datasets import load_files


def pull_data(path):
    data = load_files(path)
    return data


