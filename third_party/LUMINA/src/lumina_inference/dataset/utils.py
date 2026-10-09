"""Utility functions for processing OPF datasets.

Edge index helpers used during HeteroData construction.

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

from typing import Dict

import numpy as np
import torch
from torch import Tensor


def extract_edge_index(obj: Dict, edge_name: str) -> Tensor:
    """Extract edge index from a grid object.

    Args:
        obj (Dict): Grid object containing edges information.
        edge_name (str): Name of the edge type.

    Returns:
        Tensor: Edge index tensor.
    """
    return torch.tensor(
        np.array(
            [
                obj["grid"]["edges"][edge_name]["senders"],
                obj["grid"]["edges"][edge_name]["receivers"],
            ]
        )
    )


def extract_edge_index_rev(obj: Dict, edge_name: str) -> Tensor:
    """Extract reversed edge index from a grid object.

    Args:
        obj (Dict): Grid object containing edges information.
        edge_name (str): Name of the edge type.

    Returns:
        Tensor: Reversed edge index tensor.
    """
    return torch.tensor(
        np.array(
            [
                obj["grid"]["edges"][edge_name]["receivers"],
                obj["grid"]["edges"][edge_name]["senders"],
            ]
        )
    )
