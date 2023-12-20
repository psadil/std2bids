from std2bids.flows import ukb

from . import datasets


def test_get_ukb():
    ukb_parquet = datasets.get_ukb_parquet()
    bulk = ukb.get_ukb(ukb_parquet)
    has_expected_columns = bulk.columns == ["eid", "value"]

    assert all([has_expected_columns])


def test_get_bulk():
    ukb_parquet = datasets.get_ukb_parquet()
    d = ukb.get_ukb(ukb_parquet)
    datafields = ukb.get_bulk(d)
    eids_in = all(x in datafields.keys() for x in [1000043, 1000051])
    assert all([eids_in])
