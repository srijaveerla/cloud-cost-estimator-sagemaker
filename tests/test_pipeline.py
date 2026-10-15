import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from generate_data import make_dataset
from train import build_pipeline, CATEGORICAL, NUMERIC, TARGET


def test_model_learns_cost_signal():
    df = make_dataset(1500)
    pipe = build_pipeline(100, 3, 0.1).fit(df[CATEGORICAL + NUMERIC], df[TARGET])
    assert pipe.score(df[CATEGORICAL + NUMERIC], df[TARGET]) > 0.9


def test_spot_cheaper_than_on_demand():
    df = make_dataset(3000)
    m = df.groupby("pricing_model")[TARGET].mean()
    assert m["spot"] < m["on_demand"]
