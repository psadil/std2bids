import asyncio
import datetime
import logging
import tempfile
import typing
from pathlib import Path

import pydantic
from datalad import api as dapi
from reorganizer import convert
from reorganizer.data import ukb

from std2bids.models import abstract

Field: typing.TypeAlias = typing.Literal[
    20227,  # Functional brain images - resting - NIFTI
    20249,  # Functional brain images - task - NIFTI
    # 20250,  # Multiband diffusion brain images - NIFTI
    # 20251,  # Susceptibility weighted brain images - NIFTI
    20252,  # T1 structural brain images - NIFTI
    # 20253,  # T2 FLAIR structural brain images - NIFTI
    20263,  # T1 surface model files and additional structural segmentations (FreeSurfer)
    25750,  # rfMRI full correlation matrix, dimension 25
    25751,  # rfMRI full correlation matrix, dimension 100
    25752,  # rfMRI partial correlation matrix, dimension 25
    25753,  # rfMRI partial correlation matrix, dimension 100
    25754,  # rfMRI component amplitudes, dimension 25
    25755,  # rfMRI component amplitudes, dimension 100
    # 26300,  # Arterial spin labelling brain images - NIFTI
    # 26301,  # Quantitative susceptibility mapping images - NIFTI
    # https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=200
    31000,  # MNI Native Transform
    # https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=201
    31001,  # Native aparc a2009s dMRI
    31002,  # Native aparc dMRI
    31003,  # Native Glasser dMRI
    31004,  # Native Schaefer7n200p dMRI
    31005,  # Native Schaefer7n500p dMRI
    31006,  # Native Tian Subcortex S1 3T dMRI
    31007,  # Native Tian Subcortex S4 3T dMRI
    31008,  # Native Schaefer7n1000p dMRI
    # https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=202
    31009,  # Native aparc a2009s SF
    31010,  # Native aparc SF
    31011,  # Native Glasser SF
    31012,  # Native Schaefer7n100p to 1000p SF
    31013,  # Native Tian Subcortex S1 to S4 3T
    # https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=203
    31014,  # fMRI timeseries aparc a2009s
    31015,  # fMRI timeseries aparc
    31016,  # fMRI timeseries Glasser
    31017,  # fMRI timeseries global signal
    31018,  # fMRI timeseries Schaefer7ns 100p to 1000p
    31019,  # fMRI timeseries Tian Subcortex S1 to S4 3T
    31020,  # Connectome aparc a2009s and Tian Subcortex S1 3T
    31021,  # Connectome aparc and Tian Subcortex S1 3T
    31022,  # Connectome Glasser and Tian Subcortex S1 3T
    31023,  # Connectome Glasser and Tian Subcortex S4 3T
    31024,  # Connectome Schaefer7n1000p and Tian Subcortex S4 3T
    31025,  # Connectome Schaefer7n200p and Tian Subcortex S1 3T
    31026,  # Connectome Schaefer7n500p and Tian Subcortex S4 3T
    31027,  # Tractography endpoints coordinates
    31028,  # Tractography quality metrics
]

Instance: typing.TypeAlias = typing.Literal[2, 3]


class UKBFetcher(abstract.Fetcher):
    max_workers: int = pydantic.Field(ge=1, le=20, default=1)
    key: pydantic.FilePath
    _semaphore: asyncio.BoundedSemaphore | None = None

    def model_post_init(self, _):
        self._semaphore = asyncio.BoundedSemaphore(self.max_workers)

    @property
    def semaphore(self) -> asyncio.BoundedSemaphore:
        if not self._semaphore:
            msg = "something wrong with init"
            raise ValueError(msg)
        return self._semaphore

    async def fetch(self, bulkfile: Path, dst: Path) -> tuple[Path, Path]:
        async with self.semaphore:
            proc = await asyncio.create_subprocess_exec(
                "ukbfetch",
                f"-a{self.key}",
                f"-b{bulkfile}",
                stderr=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                cwd=dst,
            )

            stdout, stderr = await proc.communicate()

            # replacements are for Windows path requirements
            now = datetime.datetime.now().isoformat().replace(":", "-")
            stderr_file = dst / f"{now}.stderr"
            stdout_file = dst / f"{now}.stderr"
            stderr_file.write_bytes(stderr)
            stdout_file.write_bytes(stdout)

            return stderr_file, stdout_file


class DataField(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    # eid: int
    field: Field
    instance: Instance
    array: int = 0

    @property
    def extention(self) -> str:
        match self.field:
            case _ if self.field in range(25750, 25756):
                ext = "txt"
            case _:
                ext = "zip"

        return ext

    def to_str(self) -> str:
        return f"{self.field}_{self.instance}_{self.array}"

    def to_filename(self, eid: int) -> str:
        return f"{eid}_{self.to_str()}.{self.extention}"

    @classmethod
    def from_str(cls, value: str) -> "DataField":
        keys = value.split("_")
        if not len(keys) == 3:
            raise ValueError
        keys2 = [int(x) for x in keys]
        return cls(field=keys2[0], instance=keys2[1], array=keys2[2])  # type: ignore


class UKBParticipant(abstract.Participant):
    datafields: list[DataField]

    async def get_raw(self):
        old_branch = self.active_branch
        self.checkout_or_create(self.branch_incoming)

        subid_dir = Path(self.ds.path)

        with tempfile.NamedTemporaryFile() as f:
            tmpf = Path(f.name)
            bulklist: list[str] | list[typing.Never] = []
            for datafield in self.datafields:
                file_to_download = subid_dir / datafield.to_filename(
                    int(self.label)
                )
                if file_to_download.exists():
                    logging.info(
                        f"File {file_to_download} already exists. Skipping."
                    )
                else:
                    bulklist.append(f"{self.label} {datafield.to_str()}")

            if len(bulklist):
                tmpf.write_text("\n".join(bulklist))

                logging.info(f"starting eid: {self.label}")
                stdout, stderr = await self.raw_getter.fetch(
                    tmpf, dst=subid_dir
                )

                ukbatch = subid_dir / ".ukbbatch"
                if ukbatch.exists():
                    old_batch = ukbatch.read_text().splitlines()
                else:
                    old_batch = []
                (subid_dir / ".ukbbatch").write_text(
                    "\n".join(set(bulklist + old_batch))
                )
                # save
                self.repo.save(
                    paths=[ukbatch],
                    git=True,
                    message="created/updated bulk file",
                    result_renderer=None,
                )

                logging.info(f"finished downloading eid: {self.label=}")
                self.repo.save(
                    paths=[Path("fetched.lis"), stdout, stderr],
                    git=True,
                    message="adding bulk file logs",
                )

                self.repo.save(message="downloaded bulk files")
            self.repo.checkout(old_branch)

        return subid_dir

    def _raw_to_native(self):
        convert.convert_flat(
            self.path, self.path, incoming_to_natives=ukb.incoming_to_native
        )
        # extra files related to tracking download,
        # but these don't need to exist in multiple branches
        for f in self.path.glob("*std*"):
            f.unlink()
        for f in self.path.glob("fetched*"):
            f.unlink()
        for f in self.path.glob(".ukbbatch"):
            f.unlink()

    def _native_to_bids(self):
        convert.convert_recursively(
            self.path, self.path, incoming_to_natives=ukb.native_to_bids
        )

    @classmethod
    async def from_dst(
        cls,
        dst: Path,
        label: str,
        raw_getter: UKBFetcher,
        datafields: list[DataField],
        super_dataset: dapi.Dataset | None = None,
    ):
        return cls(
            label=label,
            dst=dst / f"sub-{label}",
            raw_getter=raw_getter,
            datafields=datafields,
            super_dataset=super_dataset,
        )
