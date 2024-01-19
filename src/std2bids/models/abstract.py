import abc
import shutil
import typing
from pathlib import Path

import pydantic
from datalad import api as dapi
from datalad.core.distributed import clone
from datalad.core.local import create
from datalad.distribution import siblings
from datalad.support import annexrepo

from std2bids import utils


def get_or_create_dataset(
    dst: Path, dataset: dapi.Dataset | Path | None = None
) -> dapi.Dataset:
    ds = dapi.Dataset(dst)
    if not ds.is_installed():
        ds = create.Create()(
            path=dst, dataset=dataset, initopts=["--shared=group"]
        )

    return ds


class Fetcher(pydantic.BaseModel, abc.ABC):
    @abc.abstractmethod
    def fetch(self, *args, **kwargs) -> typing.Any:
        raise NotImplementedError


class Participant(pydantic.BaseModel, abc.ABC):
    """Strategy Design for Creating Participant-level Datasets"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    label: str
    raw_getter: Fetcher

    # location for the dataset when finalized
    dst: Path

    # temporary location of the dataset when building
    _ds: dapi.Dataset | None = None
    _path: utils.TempDir = pydantic.PrivateAttr(
        default_factory=lambda: utils.TempDir()
    )

    super_dataset: dapi.Dataset | None = None

    branch_incoming: str = "incoming"
    branch_native: str = "incoming-native"
    branch_bids: str = "bids"
    branch_main: str = "main"

    def model_post_init(self, _):
        self._ds = get_or_create_dataset(self._path.path)

    @property
    def ds(self) -> dapi.Dataset:
        if not self._ds:
            msg = "something wrong with init of ds"
            raise ValueError(msg)
        return self._ds

    @property
    def path(self) -> Path:
        return self._path.path

    @property
    def repo(self) -> annexrepo.AnnexRepo:
        """Get repo associated with the ds (dataset) field

        Returns:
            Repo: The git(annex) repo

        Raises:
            ValueError: The ds did not have a repository initialized

        """
        if not self.ds.repo:
            msg = "something very strange with init"
            raise ValueError(msg)
        if not isinstance(self.ds.repo, annexrepo.AnnexRepo):
            msg = "Expected AnnexRepo"
            raise ValueError(msg)
        return self.ds.repo

    @property
    def active_branch(self) -> str:
        """Get active branch.

        Returns:
            A string with the branch name

        Raises:
            ValueError: There was no active branch
        """
        old_branch = self.repo.get_active_branch()
        if not old_branch:
            msg = "how did we end up in a detached HEAD?"
            raise ValueError(msg)
        return old_branch

    def do(self):
        # download
        self.get_raw()

        # unpack
        self.convert_raw_to_native()

        # reorganize
        self.convert_native_to_bids()

        # install in final location
        self.finalize()

    def finalize(self) -> None:
        if self.super_dataset:
            remote = f"tmp-{self.repo.uuid}"
            # need to use clone instead of install to allow use of git_clone_opts
            clone.Clone()(
                source=f"file://{self.path.absolute()}",
                path=self.dst,
                dataset=self.super_dataset,
                git_clone_opts=["--origin", f"{remote}"],
            )
            ds = dapi.Dataset(self.dst)
            if not isinstance(ds.repo, annexrepo.AnnexRepo):
                raise ValueError
            # fetch additional branches
            ds.repo._call_annex(["pull", "--all", remote], cwd=ds.repo.path)
            # ensure all branches present
            for branch in [
                self.branch_bids,
                self.branch_incoming,
                self.branch_native,
            ]:
                ds.repo.checkout(branch)
            ds.repo.checkout(self.branch_main)

            # pulling creates this weird extra branch
            ds.repo.remove_branch(f"synced/{self.branch_main}")

            # get.Get()(path=self.dst, dataset=self.super_dataset)

            # the earlier copy was only temporary, so forget it
            # need to use private method _call_annex to get cwd
            # the regular set_remote_dead operates in PWD
            ds.repo._call_annex(["dead", remote], cwd=ds.repo.path)
            siblings.Siblings()(dataset=ds, name=remote, action="remove")
        else:
            shutil.move(self.path, self.dst)

    def checkout_or_create(self, branch: str):
        """Checkout branch, or create if it doesn't exist.

        Args:
            branch: The branch to checkout (or create)
        """
        existing_branches = self.repo.get_branches()
        options = None
        if branch not in existing_branches:
            options = ["-b"]
        self.repo.checkout(branch, options=options)

    @abc.abstractmethod
    def get_raw(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def _raw_to_native(self):
        raise NotImplementedError

    def convert_raw_to_native(self):
        old_branch = self.active_branch
        self.repo.checkout(self.branch_incoming)
        self.checkout_or_create(self.branch_native)

        # main work
        self._raw_to_native()

        # save
        self.repo.save(message="unpacked bulk files")
        self.repo.checkout(old_branch)

    @abc.abstractmethod
    def _native_to_bids(self):
        raise NotImplementedError

    def convert_native_to_bids(self):
        old_branch = self.active_branch

        self.repo.checkout(self.branch_native)
        self.checkout_or_create(self.branch_bids)

        # main work
        self._native_to_bids()

        # save
        self.repo.save(message="converted unpacked files to bids-ish")

        # when moving files around, empty directories often appear
        # this gets rid of them
        self.repo.checkout(old_branch)
        self.repo.call_git(["clean", "-fd"])

        # now merge bids(is) into main brain
        self.repo.checkout(self.branch_main)
        self.repo.merge(
            self.branch_bids,
            options=[
                "-m",
                f"refreshed branch {self.branch_main} with updated {self.branch_bids}",
            ],
        )
        self.repo.checkout(old_branch)
