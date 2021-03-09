from src.datacode.retrieve_data import pull_training_data, pull_test_data


def test_load_array_size():
     file = pull_training_data('./data/raw/aclImdb/train')
     assert len(file) > 1


def test_test_array_size():
    file = pull_test_data('./data/raw/aclImdb/test')
    assert len(file) > 1

