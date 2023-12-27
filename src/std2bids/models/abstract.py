import abc
import functools
import logging
import typing
from pathlib import Path

import pydantic
from datalad import api as dapi
from datalad.support import annexrepo, gitrepo

Repo: typing.TypeAlias = gitrepo.GitRepo | annexrepo.AnnexRepo


class Fetcher(pydantic.BaseModel, abc.ABC):
    @abc.abstractmethod
    async def fetch(self, *args, **kwargs) -> typing.Any:
        raise NotImplementedError


class Participant(pydantic.BaseModel, abc.ABC):
    """Strategy Design for Creating Participant-level Datasets"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    label: str
    raw_getter: Fetcher
    ds: dapi.Dataset

    branch_incoming: str = "incoming"
    branch_native: str = "incoming-native"
    branch_bids: str = "bids"
    branch_main: str = "main"

    @functools.cached_property
    def path(self) -> Path:
        return Path(self.ds.path)

    @functools.cached_property
    def repo(self) -> Repo:
        """Get repo associated with the ds (dataset) field

        Returns:
            Repo: The git(annex) repo

        Raises:
            ValueError: The ds did not have a repository initialized

        """
        if not self.ds.repo:
            msg = "something very strange with init"
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

    async def do(self):
        # download
        await self.get_raw()

        # unpack
        await self.convert_raw_to_native()

        # reorganize
        await self.convert_native_to_bids()

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
    async def get_raw(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def _raw_to_native(self):
        raise NotImplementedError

    async def convert_raw_to_native(self):
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

    async def convert_native_to_bids(self):
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
