from config import Config, get_device


def test_config_defaults() -> None:
    c = Config(tickers=["AAPL", "MSFT"])
    assert c.n_stocks == 2
    assert c.n_features == 120
    assert c.d_model == 256
    assert c.nhead == 8
    assert c.num_layers == 4
    assert c.dim_feedforward == 512
    assert c.batch_size == 32


def test_config_no_side_effects() -> None:
    c = Config()
    assert c.tickers == []
    assert c.n_stocks == 0


def test_config_n_features() -> None:
    c = Config(features_per_window=10, n_windows=4, tickers=["A", "B"])
    assert c.n_features == 40
    assert c.n_stocks == 2


def test_get_device() -> None:
    device = get_device()
    assert str(device) in ("cuda", "mps", "cpu")
