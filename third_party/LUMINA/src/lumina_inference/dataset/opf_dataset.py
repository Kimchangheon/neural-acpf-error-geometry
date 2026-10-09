"""Customized OPFDataset class for the ACOPF task (inference only).

Supports the JSON data format produced by the public pglib-opf release.

Copyright 2026 UChicago Argonne, LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing
permissions and limitations under the License.
"""

import json
import os
import os.path as osp
import pickle
import shutil
import stat
import warnings
from glob import glob
from typing import Callable, List, Literal, Optional, Union

import torch
import tqdm
from joblib import Parallel, delayed
from torch_geometric.data import HeteroData, InMemoryDataset, download_url, extract_tar

from lumina_inference.dataset.ingestion import process_opf_dict


class OPFDataset(InMemoryDataset):
    r"""The heterogeneous OPF data from the `"Large-scale Datasets for AC
    Optimal Power Flow with Topological Perturbations"
    <https://arxiv.org/abs/2406.07234>`_ paper.

    :class:`OPFDataset` is a large-scale dataset of solved optimal power flow
    problems, derived from the
    `pglib-opf <https://github.com/power-grid-lib/pglib-opf>`_ dataset.

    The physical topology of the grid is represented by the :obj:`"bus"` node
    type, and the connecting AC lines and transformers. Additionally,
    :obj:`"generator"`, :obj:`"load"`, and :obj:`"shunt"` nodes are connected
    to :obj:`"bus"` nodes using a dedicated edge type each, *e.g.*,
    :obj:`"generator_link"`.

    Edge direction corresponds to the properties of the line, *e.g.*,
    :obj:`b_fr` is the line charging susceptance at the :obj:`from`
    (source/sender) bus.

    Args:
        root (str): Root directory where the dataset should be saved.
        case_name (str, optional): The name of the original pglib-opf case.
            (default: :obj:`"pglib_opf_case14_ieee"`)
        group_id (int, optional): The specific group to load. Each group
            contains 15,000 samples. Valid values are [0, 19].
            (default: :obj:`0`)
        topological_perturbations (bool, optional): Whether to use the dataset
            with added topological perturbations. (default: :obj:`False`)
        transform (callable, optional): A function/transform that takes in
            a :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes
            in a :obj:`torch_geometric.data.HeteroData` object and returns
            a transformed version. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :obj:`torch_geometric.data.HeteroData` object and returns a boolean
            value. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        keep_temp (bool, optional): Whether to keep the temporary files
            after processing. (default: :obj:`False`)
        n_jobs (int, optional): The number of jobs to use for parallel
            processing. If set to :obj:`-1`, all available cores will be used.
            NOTE: for larger dataset, it is recommended to set this to a lower
            positive value to avoid memory issues. (default: :obj:`-1`)
        local_raw_folder (str, optional): Local folder to look for raw files.
            (default: :obj:`None`)
    """

    url = "https://storage.googleapis.com/gridopt-dataset"

    def __init__(
        self,
        root: str,
        case_name: Literal[
            "pglib_opf_case14_ieee",
            "pglib_opf_case30_ieee",
            "pglib_opf_case57_ieee",
            "pglib_opf_case118_ieee",
            "pglib_opf_case500_goc",
            "pglib_opf_case2000_goc",
            "pglib_opf_case4661_sdet",
            "pglib_opf_case6470_rte",
            "pglib_opf_case10000_goc",
            "pglib_opf_case13659_pegase",
        ] = "pglib_opf_case14_ieee",
        group_id: int = 0,
        topological_perturbations: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        keep_temp: bool = False,
        n_jobs: int = -1,
        local_raw_folder: str = None,
    ) -> None:
        self.case_name = case_name
        self.group_id = group_id
        self.topological_perturbations = topological_perturbations

        self._raw_root = osp.join(root, "OPFData/raw")
        self._processed_root = osp.join(root, "OPFData/processed")
        self._release = "dataset_release_1"
        if topological_perturbations:
            self._release += "_nminusone"
        self.n_jobs = n_jobs
        self.keep_temp = keep_temp

        self.local_raw_folder = local_raw_folder
        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
        )

        # Load only the specified group
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

        if osp.exists(self._processed_root):
            try:
                current_mode = os.stat(self._processed_root).st_mode
                os.chmod(self._processed_root, current_mode | stat.S_IWGRP)
            except OSError as exc:
                warnings.warn(
                    f"Failed to set group write permission on "
                    f"{self._processed_root}: {exc}"
                )

    @property
    def raw_dir(self) -> str:
        """Raw data folder."""
        return osp.join(self._raw_root, self._release)

    @property
    def processed_dir(self) -> str:
        """Processed data folder."""
        return osp.join(
            self._processed_root, self._release, self.case_name
        )

    @property
    def tmp_dir(self) -> str:
        """Temporary data folder."""
        return osp.join(
            self.raw_dir,
            "gridopt-dataset-tmp",
            self._release,
            self.case_name,
        )

    @property
    def raw_file_names(self) -> List[str]:
        """Raw file names, which are stored locally."""
        return [f"{self.case_name}_{self.group_id}.tar.gz"]

    @property
    def processed_file_names(self) -> List[str]:
        """Processed file names, which are stored locally."""
        return [f"group_{self.group_id}.pt"]

    def download(self) -> None:
        """Download .tar.gz files."""
        print("Download files")
        self.download_and_extract(self.raw_file_names[0])
        print(f"Downloaded {self.raw_file_names[0]} to {self.raw_dir}")

    def download_and_extract(self, name: str) -> None:
        """Download and extract a .tar.gz file."""
        url = f"{self.url}/{self._release}/{name}"
        path = download_url(url, self.raw_dir)
        extract_tar(path, self.raw_dir)

    def process(self) -> None:
        """Process the raw JSON files into a single file."""
        if not osp.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        try:
            self.process_json_group(self.group_id)
        except Exception as e:
            print(f"Error processing group {self.group_id}: {e}")
            raise

        print(f"Processed group {self.group_id}")

        # Remove tmp_dir content to save local space
        if not self.keep_temp:
            shutil.rmtree(osp.join(self.raw_dir, "gridopt-dataset-tmp"))

    def _post_process_and_save(
        self,
        data_list: List[Optional[Union[HeteroData, List[HeteroData]]]],
        group_id: int,
    ):
        """Helper to filter, transform, collate and save processed data."""
        flattened_list = []
        for item in data_list:
            if item is None:
                continue
            if isinstance(item, list):
                flattened_list.extend(item)
            else:
                flattened_list.append(item)

        data_list = flattened_list

        if self.pre_filter is not None or self.pre_transform is not None:
            if self.pre_filter is not None:
                data_list = [
                    data for data in data_list if self.pre_filter(data)
                ]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)

        torch.save(
            (self._data, self.slices),
            osp.join(self.processed_dir, f"group_{group_id}.pt"),
        )

    def process_json_group(self, group_id: int):
        """Process a single group of JSON files, save processed data to disk.

        Args:
            group_id (int): Group id.
        """
        group_json_files = glob(
            osp.join(self.tmp_dir, f"group_{group_id}", "*.json")
        )

        if len(group_json_files) < 15000:
            extract_tar(
                osp.join(self.raw_dir, self.raw_file_names[0]), self.raw_dir
            )
            group_json_files = glob(
                osp.join(self.tmp_dir, f"group_{group_id}", "*.json")
            )

        data_list = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(process_json_file)(fn)
            for fn in tqdm.tqdm(group_json_files, desc=f"Group {group_id}")
        )

        self._post_process_and_save(data_list, group_id)

    def metadata(self):
        """Returns the metadata of the dataset."""
        return {
            "nodes": {
                "bus": self._data["bus"].x.size(1),
                "generator": self._data["generator"].x.size(1),
                "load": self._data["load"].x.size(1),
                "shunt": self._data["shunt"].x.size(1),
            },
            "edges": {
                ("bus", "ac_line", "bus"): self._data[
                    "bus", "ac_line", "bus"
                ].edge_attr.size(1),
                ("bus", "transformer", "bus"): self._data[
                    "bus", "transformer", "bus"
                ].edge_attr.size(1),
                ("generator", "generator_link", "bus"): 0,
                ("bus", "generator_link", "generator"): 0,
                ("load", "load_link", "bus"): 0,
                ("bus", "load_link", "load"): 0,
                ("shunt", "shunt_link", "bus"): 0,
                ("bus", "shunt_link", "shunt"): 0,
            },
        }

    def __repr__(self) -> str:
        """Returns the string representation of the dataset."""
        return (
            f"{self.__class__.__name__}({len(self)}, "
            f"case_name={self.case_name}, "
            f"topological_perturbations={self.topological_perturbations})"
        )


# ---------------------------------------------------------------------------
# JSON processing
# ---------------------------------------------------------------------------


def process_json_file(json_file, require_solution=True):
    """Process a single JSON file into a HeteroData object.

    This function delegates to :func:`~lumina_inference.dataset.ingestion.process_opf_dict`
    for the actual conversion. It can be used both for training (with solution
    data) and for inference (without solution data).

    Args:
        json_file (str): Path to the JSON file.
        require_solution (bool): If True (default), the JSON must contain
            a ``'solution'`` key. Set to False for inference-only usage
            where ground-truth solutions are not available.

    Returns:
        HeteroData or None: Processed data object, or None if JSON
        decoding fails.
    """
    with open(json_file) as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {json_file}")
            return None

    return process_opf_dict(obj, require_solution=require_solution)
