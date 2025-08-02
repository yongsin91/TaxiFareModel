import pandas as pd
from TaxiFareModel import data as data_module


def test_get_data_respects_nrows(tmp_path, monkeypatch):
    file_path = tmp_path / 'train.csv'
    df = pd.DataFrame({'a': range(10)})
    df.to_csv(file_path, index=False)
    monkeypatch.setattr(data_module, 'FILE_PATH', file_path)
    loaded = data_module.get_data(nrows=5)
    assert len(loaded) == 5


def test_get_data_respects_nrows_on_test_file(tmp_path, monkeypatch):
    file_path = tmp_path / 'test.csv'
    df = pd.DataFrame({'a': range(20)})
    df.to_csv(file_path, index=False)
    monkeypatch.setattr(data_module, 'TEST_FILE_PATH', file_path)
    loaded = data_module.get_data(nrows=7, test=True)
    assert len(loaded) == 7
