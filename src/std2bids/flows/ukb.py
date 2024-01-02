import argparse
import asyncio
import logging
import typing
from pathlib import Path

import polars as pl
import polars.selectors as s
from datalad import api as dapi

from std2bids.models import ukb as ukb_models

# TODO: confirm that updates will work as expected (download new files, unpack, bidsify, keep clean history)


def get_bulk(d: pl.DataFrame) -> dict[int, list[ukb_models.DataField]]:
    """Convert UKB data into a format closer to the bulk file format

    Args:
        d: The biobank in a long format. Each row should be a participant, with
            the bulk fields stored in column "value", whose entries are lists of strings

    Returns:
        dict: The same data. Each key is a participant and each value is a list of the bulk
        fields that are available for that participant.
    """
    datafields = {}
    for eid in d.to_dicts():
        bulk = []
        for datafield in eid.get("value", []):
            bulk.append(ukb_models.DataField.from_str(datafield))
        datafields.update({eid.get("eid"): bulk})
    return datafields


def get_ukb(ukb: Path) -> pl.DataFrame:
    fields = [s.starts_with(str(x)) for x in typing.get_args(ukb_models.Field)]
    d = (
        pl.scan_parquet(ukb)
        .select("eid", *fields)
        .filter(pl.col("20252-2.0").is_not_null())
        .melt(id_vars="eid")
        .drop_nulls()
        .select("eid", "value")
        .group_by("eid")
        .all()
        .sort("eid")
        .collect()
    )
    return d


async def flow(
    bulk: Path,
    dst: Path,
    key: Path,
    max_workers: int = 1,
    do_participants: bool = True,
    super_dataset: dapi.Dataset | None = None,
    max_participants: float = float("inf"),
    shortcut: bool = False,
):
    """Main workflow

    Args:
        bulk (Path): _description_
        dst (Path): _description_
        key (Path): _description_
        max_workers (int, optional): _description_. Defaults to 1.
        do_participants (bool, optional): _description_. Defaults to True.
        super_dataset (dapi.Dataset | None, optional): _description_. Defaults to None.
        super_dataset (int): _description_. Defaults to 1.
        shortcut (bool, optional): _description_. Defaults to False.
    """
    d = get_ukb(bulk)
    datafields = get_bulk(d)
    fetcher = ukb_models.UKBFetcher(max_workers=max_workers, key=key)

    async with asyncio.TaskGroup() as tg:
        for i, (eid, eid_datafields) in enumerate(datafields.items()):
            if (dst / f"sub-{eid}").exists() and shortcut:
                logging.warning(f"shortcutting {i=}, {eid=}")
            else:
                logging.info(f"starting {i=}, {eid=}")
                participant = await tg.create_task(
                    ukb_models.UKBParticipant.from_dst(
                        dst=dst,
                        label=str(eid),
                        raw_getter=fetcher,
                        datafields=eid_datafields,
                        super_dataset=super_dataset,
                    )
                )
                tg.create_task(participant.do())

            if i >= max_participants:
                break

    if do_participants:
        d.with_columns(
            participant_label=pl.concat_str(
                pl.Series(["sub-"]), pl.col("eid")
            ),
            fields=pl.col("value").list.join(separator=","),
        ).drop(["eid", "value"]).write_csv(
            dst / "participants.tsv", separator="\t", null_value="n/a"
        )


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("bulk", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("key", type=Path)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument(
        "--super-dataset",
        type=Path,
        help="Superdataset underneath into which datasets will be installed",
    )
    parser.add_argument(
        "--do-participants",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also create a bids compliant participants file",
    )
    parser.add_argument(
        "--max-participants",
        default=float("inf"),
        type=float,
        help="maximum number of participants to download (helpful for testing)",
    )
    parser.add_argument(
        "--shortcut",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to skip entirely participants that already have a directory",
    )

    args = parser.parse_args()
    if args.max_workers > 20:
        msg = "UKB does not allow more than 20 simultaneous connections"
        raise ValueError(msg)

    if args.super_dataset:
        super_dataset = dapi.Dataset(args.super_dataset)
        if not super_dataset.is_installed():
            msg = "The superdataset must already be installed."
            raise ValueError(msg)
    else:
        super_dataset = None

    asyncio.run(
        flow(
            bulk=args.bulk,
            dst=args.destination,
            key=args.key,
            max_workers=args.max_workers,
            do_participants=args.do_participants,
            super_dataset=super_dataset,
            max_participants=args.max_participants,
            shortcut=args.shortcut,
        )
    )


if __name__ == "__main__":
    main()
