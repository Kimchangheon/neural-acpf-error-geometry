"""Inference Modeler — load a trained LUMINA model and run predictions.

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

import ast
import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from lumina_inference.dataset.validation import (
    MissingDataError,
    SchemaValidationError,
    detect_schema,
    validate_hetero_data,
)
from lumina_inference.model.hetero_model import HGT


def _safe_edge_attr_dict(batch):
    """Return edge_attr_dict if any edges carry attributes, else None.

    HeteroData.edge_attr_dict is a property that raises KeyError when no
    edge type has an edge_attr tensor, so a plain hasattr check is not
    sufficient.
    """
    try:
        return batch.edge_attr_dict
    except KeyError:
        return None


class Modeler:
    """Prediction-only wrapper for OPF model loading and inference.

    This class encapsulates model construction from a Hugging Face config,
    weight loading from SafeTensors state dicts, and batch prediction.

    Args:
        device (torch.device): Device to run model inference on.
        fail_on_missing (bool, optional): If True, raise when expected model
            keys are missing during checkpoint load. Defaults to False.
        verbose (bool, optional): If True, print diagnostic messages during
            checkpoint loading. Defaults to True.

    Attributes:
        device (torch.device): See Args.
        fail_on_missing (bool): See Args.
        verbose (bool): See Args.
        model (Optional[torch.nn.Module]): Loaded model (None until
            ``load_model`` is called).
        config_data (Optional[dict]): Parsed model configuration loaded
            during ``load_model``.

    Example:
        >>> modeler = Modeler(torch.device("cpu"))
        >>> config = json.load(open("config.json"))
        >>> state_dict = load_file("model.safetensors")
        >>> modeler.load_model(config, state_dict)
        >>> loader = DataLoader(dataset, batch_size=1)
        >>> preds = modeler.run_predictions(loader, max_batches=10)
    """

    def __init__(
        self,
        device: torch.device,
        *,
        fail_on_missing: bool = False,
        verbose: bool = True,
    ):
        self.device = device
        self.fail_on_missing = fail_on_missing
        self.verbose = verbose
        self.model: Optional[torch.nn.Module] = None
        self.config_data = None

    # -- checkpoint key conversion and loading --------------------------------

    @staticmethod
    def convert_checkpoint_key_to_model_key(key: str) -> str:
        """Convert checkpoint keys to model keys.

        Transforms underscore-delimited items inside angle brackets to
        tuple string representation.

        Args:
            key: Current key with triple underscore delimiters inside
                angle brackets.

        Returns:
            String with angle bracket contents converted to tuple
            representation.

        Example:
            >>> Modeler.convert_checkpoint_key_to_model_key(
            ...     "<bus___ac_line___weight>"
            ... )
            "('bus', 'ac_line', 'weight')"
        """
        pattern = r"<([^>]+)>"

        def replacer(match):
            parts = match.group(1).split("___")
            return f"('{parts[0]}', '{parts[1]}', '{parts[2]}')"

        return re.sub(pattern, replacer, key)

    def load_checkpoint_into_model(
        self,
        model: torch.nn.Module,
        checkpoint_dict,
        *,
        fail_on_missing: bool = False,
        verbose: bool = True,
    ):
        """Load a checkpoint dictionary into a model.

        Remaps checkpoint keys to model keys using
        ``convert_checkpoint_key_to_model_key`` and then calls
        ``load_state_dict`` with ``strict=False``.

        Args:
            model (torch.nn.Module): The model to populate.
            checkpoint_dict (dict): Mapping of checkpoint keys to tensors.
            fail_on_missing (bool, optional): If True, raise ValueError
                when missing keys remain. Defaults to False.
            verbose (bool, optional): If True, print missing/unexpected keys.

        Returns:
            dict: A dictionary with keys ``"missing_keys"`` and
            ``"unexpected_keys"``.

        Raises:
            ValueError: If ``fail_on_missing`` is True and missing keys
                are found.
        """
        model_state = model.state_dict()
        used_keys = set()

        remapped_state = {}
        for model_key in model_state.keys():
            ck = self.convert_checkpoint_key_to_model_key(model_key)
            if ck in checkpoint_dict:
                remapped_state[model_key] = checkpoint_dict[ck]
                used_keys.add(ck)

        unexpected_keys = [
            k for k in checkpoint_dict.keys() if k not in used_keys
        ]

        load_result = model.load_state_dict(remapped_state, strict=False)
        missing_keys = list(load_result.missing_keys)
        unexpected_keys.extend(list(load_result.unexpected_keys))

        if verbose and (missing_keys or unexpected_keys):
            print(
                f"[CHECKPOINT LOAD] Missing keys: {missing_keys}, "
                f"Unexpected keys: {unexpected_keys}"
            )
        if fail_on_missing and missing_keys:
            raise ValueError(f"Missing keys during load: {missing_keys}")

        return {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }

    # -- model loading --------------------------------------------------------

    def load_model(self, config_data: dict, state_dict: dict):
        """Construct the HGT model from config and state dict.

        Args:
            config_data (dict): Parsed JSON configuration describing model
                metadata and architecture.
            state_dict (dict): Raw state dictionary as returned by
                ``safetensors.torch.load_file``.

        Returns:
            Tuple[torch.nn.Module, dict]: The constructed model (in eval
            mode) and the config_data used to build it.

        Raises:
            ValueError: If ``fail_on_missing`` is True and required keys
                are missing from the checkpoint.
        """
        # Convert metadata edge keys from strings to tuples if needed
        if "edges" in config_data.get("metadata", {}):
            edges_dict = {}
            for key, value in config_data["metadata"]["edges"].items():
                if isinstance(key, str) and key.startswith("("):
                    key = ast.literal_eval(key)
                edges_dict[key] = value
            config_data["metadata"]["edges"] = edges_dict

        models_cfg = config_data["config"]["models"]
        hgt_cfg = models_cfg.get("HGT") or models_cfg.get("HeteroGNN")
        if hgt_cfg is None:
            raise KeyError(
                "config['models'] must contain an 'HGT' or 'HeteroGNN' key"
            )
        model = HGT(
            metadata=config_data["metadata"],
            input_channels=config_data["input_channels"],
            hidden_channels=hgt_cfg["hidden_channels"],
            num_layers=hgt_cfg["num_layers"],
            num_heads=hgt_cfg.get("num_heads", 1),
            dropout=hgt_cfg.get("dropout", 0.0),
        ).to(self.device)

        # state_dict is the raw output of safetensors.load_file; remap keys
        checkpoint_dict = {
            self.convert_checkpoint_key_to_model_key(k): v
            for k, v in state_dict.items()
        }

        self.load_checkpoint_into_model(
            model,
            checkpoint_dict,
            fail_on_missing=self.fail_on_missing,
            verbose=self.verbose,
        )

        model.eval()
        self.model = model
        self.config_data = config_data
        return model, config_data

    # -- helpers for tensors --------------------------------------------------

    @staticmethod
    def to_float32(batch):
        """Convert node features/targets and edge attributes to float32.

        Iterates through all node types and edge types in the batch,
        converting ``x`` (features), ``y`` (targets), and ``edge_attr``
        tensors to float32 precision.

        Args:
            batch: A batch data object from DataLoader.

        Returns:
            The same batch object with tensors converted to float32.
        """
        for node_type in batch.node_types:
            if getattr(batch[node_type], "x", None) is not None:
                batch[node_type].x = batch[node_type].x.float()
            if getattr(batch[node_type], "y", None) is not None:
                batch[node_type].y = batch[node_type].y.float()

        for edge_type in batch.edge_types:
            if getattr(batch[edge_type], "edge_attr", None) is not None:
                batch[edge_type].edge_attr = batch[edge_type].edge_attr.float()

        return batch

    # -- prediction -----------------------------------------------------------

    def predict_batch(self, batch, minmax_scaling: bool = True):
        """Run a forward pass on a single batch.

        Args:
            batch: Batched data object containing inputs for the model.
            minmax_scaling (bool, optional): Whether to apply min-max
                scaling in the model's forward pass.

        Returns:
            Tuple[dict, object]: A tuple ``(predictions_cpu, batch_cpu)``
            where ``predictions_cpu`` maps output names to CPU tensors
            and ``batch_cpu`` is the input batch moved to CPU.

        Raises:
            RuntimeError: If the model has not been loaded via
                ``load_model()``.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        batch = self.to_float32(batch).to(self.device)

        predictions = self.model(
            batch.x_dict,
            batch.edge_index_dict,
            _safe_edge_attr_dict(batch),
            minmax_scaling=minmax_scaling,
        )

        # Move predictions to CPU and detach
        predictions_cpu = {}
        for k, v in predictions.items():
            if isinstance(v, torch.Tensor):
                predictions_cpu[k] = v.detach().cpu()
            else:
                predictions_cpu[k] = v

        batch_cpu = batch.to(torch.device("cpu"))
        return predictions_cpu, batch_cpu

    def run_predictions(
        self,
        loader: Iterable,
        max_batches: Optional[int] = None,
        minmax_scaling: bool = True,
    ):
        """Run predictions over a data loader.

        This method separates the forward pass from evaluation so
        predictions can be stored or evaluated later.

        Args:
            loader (Iterable): Iterable data loader yielding batches.
            max_batches (Optional[int], optional): Limit on number of
                batches to process. Defaults to None (process all).
            minmax_scaling (bool, optional): Passed to ``predict_batch``.
                Defaults to True.

        Returns:
            List[Tuple[dict, object]]: List of
            ``(predictions_cpu, batch_cpu)`` tuples.

        Raises:
            RuntimeError: If the model has not been loaded via
                ``load_model()``.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        pred_batch_pairs: List[Tuple[dict, object]] = []
        total_batches = None
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        if total_batches is not None and max_batches is not None:
            total_batches = min(total_batches, max_batches)

        progress_iter = tqdm(
            loader, total=total_batches, desc="Predicting samples"
        )
        for batch_idx, batch in enumerate(progress_iter):
            preds, batch_cpu = self.predict_batch(
                batch, minmax_scaling=minmax_scaling
            )
            pred_batch_pairs.append((preds, batch_cpu))

            progress_iter.set_postfix(
                predictions=len(pred_batch_pairs), refresh=False
            )
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                progress_iter.write(f"Reached max_batches={max_batches}.")
                break
        progress_iter.close()
        return pred_batch_pairs

    # -- single-sample prediction ---------------------------------------------

    def predict_single(
        self,
        data: HeteroData,
        minmax_scaling: bool = True,
        validate: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run inference on a single HeteroData object.

        Wraps the data in a batch of size 1 and delegates to
        :meth:`predict_batch`. This is the primary entry point for
        flexible ingestion — users can pass HeteroData from any of the
        ``load_from_*`` functions or :func:`build_hetero_data`.

        Args:
            data (HeteroData): A single heterogeneous graph data object.
            minmax_scaling (bool, optional): Whether to apply min-max
                scaling in the model's forward pass. Defaults to True.
            validate (bool, optional): Whether to validate the data
                against the loaded model's schema before prediction.
                Defaults to True.

        Returns:
            Dict[str, torch.Tensor]: Predictions keyed by output node
            type (e.g., ``'bus'``, ``'generator'``), with tensors on CPU.

        Raises:
            RuntimeError: If the model has not been loaded via
                ``load_model()``.
            SchemaValidationError: If validation is enabled and the data
                doesn't match the model's expected schema.
            FeatureDimensionError: If validation is enabled and feature
                dimensions don't match.
            MissingDataError: If validation is enabled and required
                tensors are missing.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if validate:
            self._validate_batch(data)

        from torch_geometric.data import Batch

        batch = Batch.from_data_list([data])
        predictions_cpu, _ = self.predict_batch(
            batch, minmax_scaling=minmax_scaling
        )
        return predictions_cpu

    # -- validation -----------------------------------------------------------

    def _validate_batch(self, data: HeteroData) -> None:
        """Validate a HeteroData object against the loaded model's schema.

        Checks that the data has all node types and feature dimensions
        expected by the model. Uses the model's ``config_data`` to
        determine expected types and dimensions.

        Args:
            data (HeteroData): Data to validate.

        Raises:
            RuntimeError: If the model has not been loaded.
            SchemaValidationError: If structural requirements are not met.
            FeatureDimensionError: If feature dimensions don't match.
            MissingDataError: If required tensors are missing.
        """
        if self.model is None or self.config_data is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        model_metadata = self.config_data.get("metadata")
        model_input_channels = self.config_data.get("input_channels")

        validate_hetero_data(
            data,
            model_metadata=model_metadata,
            model_input_channels=model_input_channels,
        )
