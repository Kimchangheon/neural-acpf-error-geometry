"""Heterogeneous Graph Transformer model for ACOPF inference.

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

import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class HGT(torch.nn.Module):

    def __init__(self,
                 metadata,
                 input_channels,
                 hidden_channels=64,
                 out_channels=2,
                 num_layers=3,
                 num_heads=1,
                 dropout=0.0,
                 backend="sage",
                 edge_attr_dim=None,
                 **kwargs):
        r""" Heterogeneous Graph Transformer (HGT) model.

        Args:
            metadata (dict or tuple): Metadata containing node types and edge types.
                If dict: {'nodes': {node_type: dim, ...}, 'edges': {edge_type: dim, ...}}
                If tuple: (node_types, edge_types)
            input_channels (dict): Number of input features for each node type.
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample. Defaults to 2.
            num_layers (int): Number of layers. Defaults to 3.
            num_heads (int): Number of multi-head-attention heads. Defaults to 1.
            backend (str): Graph convolutional layer backend. Defaults to "sage".
            edge_attr_dim (int, optional): Dimension of edge attributes for GAT. Defaults to None.
        """
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        self.dropout = float(dropout)

        # Handle both old tuple format and new dict format
        if isinstance(metadata, dict):
            self.node_types = list(metadata['nodes'].keys())
            self.edge_types = list(metadata['edges'].keys())
        else:
            # Legacy tuple/list format: (node_types, edge_types)
            self.node_types = list(metadata[0])
            self.edge_types = [
                tuple(et) if isinstance(et, list) else et for et in metadata[1]
            ]

        # Validate input_channels
        if not isinstance(input_channels, dict):
            raise ValueError("input_channels must be a dictionary")

        for node_type in self.node_types:
            if node_type not in input_channels:
                raise ValueError(f"input_channels must contain entry for node type '{node_type}'")
            self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)

        metadata_tuple = (self.node_types, self.edge_types)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata_tuple, num_heads)
            self.convs.append(conv)

        # Output layers for target node types
        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of the model."""
        for lin in self.lin_dict.values():
            lin.reset_parameters()
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        for out in self.out_dict.values():
            out.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, minmax_scaling=False, **kwargs):
        """Forward pass of the HGT model.

        Args:
            x_dict (dict): Node features for each node type.
            edge_index_dict (dict): Edge indices for each edge type.
            edge_attr_dict (dict, optional): Edge attributes for each edge type. Defaults to None
            minmax_scaling (bool): Whether to apply min-max scaling to outputs. Defaults to False.

        Returns:
            dict: Output predictions for each target node type.
        """

        if minmax_scaling:
            _vmin = x_dict['bus'][:, 1].clone()  # Original voltage min
            _vmax = x_dict['bus'][:, 2].clone()  # Original voltage max
            _pmin = x_dict['generator'][:, 2].clone()  # Original active power min
            _pmax = x_dict['generator'][:, 3].clone()  # Original active power max
            _qmin = x_dict['generator'][:, 5].clone()  # Original reactive power min
            _qmax = x_dict['generator'][:, 6].clone()  # Original reactive power max

        # Transform input features
        x_dict = {
            node_type: torch.relu(self.lin_dict[node_type](x_dict[node_type]))
            for node_type in self.node_types
        }
        if self.dropout > 0.0:
            x_dict = {
                key: F.dropout(x, p=self.dropout, training=self.training)
                for key, x in x_dict.items()
            }

        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            if self.dropout > 0.0:
                x_dict = {
                    key: F.dropout(x, p=self.dropout, training=self.training)
                    for key, x in x_dict.items()
                }

        # Final predictions
        bus_out = self.out_dict["bus"](x_dict["bus"])
        gen_out = self.out_dict["generator"](x_dict["generator"])

        if minmax_scaling:
            bus_out_final = bus_out.clone()
            gen_out_final = gen_out.clone()

            bus_out_final[:, 1] = F.sigmoid(bus_out[:, 1]) * (_vmax - _vmin) + _vmin

            gen_out_sigmoid = F.sigmoid(gen_out)
            gen_out_final[:, 0] = gen_out_sigmoid[:, 0] * (_pmax - _pmin) + _pmin
            gen_out_final[:, 1] = gen_out_sigmoid[:, 1] * (_qmax - _qmin) + _qmin

            return {"bus": bus_out_final, "generator": gen_out_final}
        else:
            return {"bus": bus_out, "generator": gen_out}
