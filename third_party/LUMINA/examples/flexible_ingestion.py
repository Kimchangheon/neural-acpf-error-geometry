"""Example: Flexible ingestion paths for LUMINA inference.

This script demonstrates all the ways to load data into LUMINA for inference,
**including end-to-end prediction** with a loaded model:

 1. Load from a Python dictionary → predict
 2. Load from a JSON string → predict
 3. Load from a JSON file (process_json_file) → predict
 4. Load from a MATPOWER .m file → predict  (requires ``pip install pandapower``)
 5. Build generic HeteroData from raw arrays → predict
 6. Schema validation utilities

Usage:
    python examples/flexible_ingestion.py

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
import tempfile

import numpy as np
import torch
from torch_geometric.data import HeteroData

# Import ingestion functions from lumina-inference
from lumina_inference import (
    Modeler,
    build_hetero_data,
    load_from_dict,
    load_from_json_file,
    load_from_json_string,
    process_json_file,
    process_opf_dict,
)
from lumina_inference.dataset.validation import (
    detect_schema,
    validate_hetero_data,
    validate_opf_schema,
)


# ============================================================================
# Helper: Create a minimal OPF data dictionary for demonstration
# ============================================================================

def make_sample_opf_dict(with_solution=False):
    """Create a minimal OPF data dictionary (IEEE 14-bus-like).

    This is a simplified example. Real OPF data would come from
    power system simulation tools.
    """
    opf_dict = {
        "grid": {
            "context": [[[100.0]]],  # baseMVA
            "nodes": {
                # bus: [base_kv, bus_type, vmin, vmax]
                "bus": [
                    [69.0, 3, 0.94, 1.06],   # bus 0 (ref)
                    [69.0, 2, 0.94, 1.06],   # bus 1 (PV)
                    [69.0, 1, 0.94, 1.06],   # bus 2 (PQ)
                    [69.0, 1, 0.94, 1.06],   # bus 3 (PQ)
                ],
                # generator: [mbase, pg, pmin, pmax, qg, qmin, qmax, vg,
                #              cost_sq, cost_lin, cost_off]
                "generator": [
                    [100.0, 2.324, 0.0, 3.324, 0.0, -0.1, 0.1, 1.06,
                     0.043, 20.0, 0.0],
                    [100.0, 0.40, 0.0, 1.40, 0.0, -0.4, 0.5, 1.045,
                     0.25, 20.0, 0.0],
                ],
                # load: [pd, qd]
                "load": [
                    [0.217, 0.127],
                    [0.942, 0.190],
                    [0.478, -0.039],
                ],
                # shunt: [bs, gs]
                "shunt": [
                    [0.19, 0.0],
                ],
            },
            "edges": {
                "ac_line": {
                    "senders": [0, 1, 1, 2],
                    "receivers": [1, 2, 3, 3],
                    "features": [
                        [-0.5236, 0.5236, 0.0264, 0.0264, 0.01938, 0.05917, 4.72, 4.72, 4.72],
                        [-0.5236, 0.5236, 0.0219, 0.0219, 0.05403, 0.22304, 1.28, 1.28, 1.28],
                        [-0.5236, 0.5236, 0.0170, 0.0170, 0.04699, 0.19797, 1.28, 1.28, 1.28],
                        [-0.5236, 0.5236, 0.0128, 0.0128, 0.06701, 0.17103, 0.65, 0.65, 0.65],
                    ],
                },
                "transformer": {
                    "senders": [0],
                    "receivers": [3],
                    "features": [
                        [-0.5236, 0.5236, 0.0, 0.20912, 0.978, 0.978, 0.978, 0.969, 0.0, 0.0, 0.0],
                    ],
                },
                "generator_link": {
                    "senders": [0, 1],
                    "receivers": [0, 1],
                },
                "load_link": {
                    "senders": [0, 1, 2],
                    "receivers": [1, 2, 3],
                },
                "shunt_link": {
                    "senders": [0],
                    "receivers": [3],
                },
            },
        },
        "metadata": {"objective": 0.0},
    }

    if with_solution:
        opf_dict["solution"] = {
            "nodes": {
                "bus": [
                    [0.0, 1.06],
                    [-0.0869, 1.045],
                    [-0.2217, 1.010],
                    [-0.1800, 1.018],
                ],
                "generator": [
                    [2.324, -0.169],
                    [0.40, 0.424],
                ],
            },
            "edges": {
                "ac_line": {
                    "features": [
                        [1.569, 0.0, -1.569, 0.0],
                        [0.732, 0.0, -0.732, 0.0],
                        [0.243, 0.0, -0.243, 0.0],
                        [0.282, 0.0, -0.282, 0.0],
                    ],
                },
                "transformer": {
                    "features": [
                        [0.755, 0.0, -0.755, 0.0],
                    ],
                },
            },
        }

    return opf_dict


# ============================================================================
# Helper: Initialize a Modeler (from HuggingFace or local checkpoint)
# ============================================================================

def init_modeler(device=None):
    """Initialize a Modeler from HuggingFace Hub.

    Returns None if the model cannot be loaded (e.g. no network, no auth).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        modeler = Modeler(device)
        modeler.from_pretrained("argonne/LUMINA-OPF-HGNN")
        return modeler
    except Exception as e:
        print(f"  [WARN] Could not load model: {e}")
        print("  Skipping live prediction — showing code pattern only.")
        return None


# ============================================================================
# Helper: Run prediction and display results
# ============================================================================

def show_prediction(modeler, data, label=""):
    """Run predict_single and print result shapes."""
    if modeler is None:
        print(f"  {label} → (model not loaded, skipping prediction)")
        return

    predictions = modeler.predict_single(data)
    for node_type, pred_tensor in predictions.items():
        print(f"  {label} → {node_type}: {pred_tensor.shape}")


# ============================================================================
# Example 1: Load from a Python dictionary → predict
# ============================================================================

print("=" * 70)
print("Example 1: Python dict → HeteroData → predict_single()")
print("=" * 70)

opf_dict = make_sample_opf_dict(with_solution=False)
data = load_from_dict(opf_dict, require_solution=False)

print(f"  Schema detected : {detect_schema(data)}")
print(f"  Node types      : {data.node_types}")
print(f"  Edge types      : {data.edge_types}")
print(f"  Bus features    : {data['bus'].x.shape}")
print(f"  Gen features    : {data['generator'].x.shape}")
print(f"  Has bus.y       : {hasattr(data['bus'], 'y') and data['bus'].y is not None}")

modeler = init_modeler()
show_prediction(modeler, data, label="dict→predict")
print()


# ============================================================================
# Example 2: Load from a JSON string → predict
# ============================================================================

print("=" * 70)
print("Example 2: JSON string → HeteroData → predict_single()")
print("=" * 70)

json_str = json.dumps(make_sample_opf_dict(with_solution=False))
data_from_str = load_from_json_string(json_str, require_solution=False)

print(f"  Schema detected : {detect_schema(data_from_str)}")
print(f"  Bus features    : {data_from_str['bus'].x.shape}")

show_prediction(modeler, data_from_str, label="json_str→predict")
print()


# ============================================================================
# Example 3: Load from a JSON file (process_json_file) → predict
# ============================================================================

print("=" * 70)
print("Example 3: JSON file → HeteroData → predict_single()")
print("=" * 70)

opf_dict_with_sol = make_sample_opf_dict(with_solution=True)
with tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", delete=False
) as f:
    json.dump(opf_dict_with_sol, f)
    tmp_json = f.name

try:
    # With solution (backward compatible)
    data_from_file = process_json_file(tmp_json, require_solution=True)
    print(f"  With solution   : bus.y = {data_from_file['bus'].y.shape}")

    # Without solution (inference-only mode)
    data_no_sol = process_json_file(tmp_json, require_solution=False)
    print(f"  Without solution: loaded OK, bus.x = {data_no_sol['bus'].x.shape}")

    show_prediction(modeler, data_no_sol, label="json_file→predict")
finally:
    os.unlink(tmp_json)
print()


# ============================================================================
# Example 4: Load from MATPOWER .m file → predict  (optional dependency)
# ============================================================================

print("=" * 70)
print("Example 4: MATPOWER .m file → HeteroData → predict_single()")
print("=" * 70)

print("  Requires pandapower: pip install pandapower")
print()
print("    from lumina_inference import load_from_matpower")
print()
print("    data = load_from_matpower('path/to/case14.m')")
print("    print(detect_schema(data))  # 'opf'")
print("    predictions = modeler.predict_single(data)")
print("    print(predictions['bus'].shape)       # [num_buses, 2]")
print("    print(predictions['generator'].shape) # [num_gens, 2]")
print()


# ============================================================================
# Example 5: Build generic HeteroData (non-OPF) → predict
# ============================================================================

print("=" * 70)
print("Example 5: Generic HeteroData (non-OPF) → predict_single()")
print("=" * 70)

generic_data = build_hetero_data(
    nodes={
        "station": {"x": np.random.randn(5, 8).astype(np.float32)},
        "sensor": {"x": np.random.randn(10, 4).astype(np.float32)},
    },
    edges={
        ("station", "connects", "station"): {
            "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
            "edge_attr": torch.randn(4, 3),
        },
        ("sensor", "monitors", "station"): {
            "edge_index": torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]]
            ),
        },
    },
    graph_attrs={"name": "sensor_network", "timestamp": 1234567890},
)

print(f"  Schema detected : {detect_schema(generic_data)}")
print(f"  Node types      : {generic_data.node_types}")
print(f"  Edge types      : {generic_data.edge_types}")
print(f"  Station features: {generic_data['station'].x.shape}")
print(f"  Sensor features : {generic_data['sensor'].x.shape}")
print(f"  Graph attr      : name={generic_data.name}")
print()
print("  NOTE: Generic graphs use the model's forward() without OPF-specific")
print("  minmax scaling. The model must have been trained on a compatible schema.")
print()


# ============================================================================
# Example 6: Schema validation
# ============================================================================

print("=" * 70)
print("Example 6: Schema validation utilities")
print("=" * 70)

# Validate OPF data
opf_data = load_from_dict(make_sample_opf_dict(), require_solution=False)
try:
    validate_opf_schema(opf_data)
    print("  OPF schema validation: PASSED")
except Exception as e:
    print(f"  OPF schema validation: FAILED - {e}")

# Validate generic data against model metadata
try:
    validate_hetero_data(
        generic_data,
        model_metadata={"nodes": {"station": 8, "sensor": 4}},
        model_input_channels={"station": 8, "sensor": 4},
    )
    print("  Generic data validation: PASSED")
except Exception as e:
    print(f"  Generic data validation: FAILED - {e}")

# Demonstrate validation failure
try:
    validate_hetero_data(
        generic_data,
        model_input_channels={"station": 16},  # Wrong dimension
    )
    print("  Wrong dimension validation: PASSED (unexpected)")
except Exception as e:
    print(f"  Wrong dimension validation: FAILED (expected) - {type(e).__name__}")
print()


# ============================================================================
# End-to-end summary
# ============================================================================

print("=" * 70)
print("End-to-end prediction pattern (all ingestion paths)")
print("=" * 70)
print("""
  # 1. Initialize the modeler (once)
  modeler = Modeler(torch.device('cpu'))
  modeler.from_pretrained('argonne/LUMINA-OPF-HGNN')

  # 2. Load data from ANY source
  data = load_from_dict(opf_dict)          # Python dict
  data = load_from_json_string(json_str)   # JSON string
  data = load_from_json_file('data.json')  # JSON file
  data = load_from_matpower('case14.m')    # MATPOWER file
  data = build_hetero_data(nodes, edges)   # Raw arrays

  # 3. Predict (single sample — no DataLoader needed)
  predictions = modeler.predict_single(data)
  bus_va_vm = predictions['bus']           # [num_buses, 2]  (va, vm)
  gen_pg_qg = predictions['generator']    # [num_gens, 2]   (pg, qg)

  # 4. Or batch predict with a DataLoader
  from torch_geometric.loader import DataLoader
  loader = DataLoader([data1, data2, data3], batch_size=2)
  all_preds = modeler.run_predictions(loader)
""")

print("=" * 70)
print("All examples completed successfully!")
print("=" * 70)
