from pathlib import Path

DATA_PATH = Path(__file__).parent / "data"


def get_ukb_parquet() -> Path:
    return DATA_PATH / "ukb-small.parquet"
