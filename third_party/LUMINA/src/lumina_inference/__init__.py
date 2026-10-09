"""
lumina-inference: Lightweight inference package for LUMINA trained models.

Load models from Hugging Face and run predictions.

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

__version__ = "0.1.0"

from lumina_inference.dataset.ingestion import (
    build_hetero_data,
    load_from_dict,
    load_from_json_file,
    load_from_json_string,
    load_from_matpower,
    process_opf_dict,
)
from lumina_inference.dataset.opf_dataset import process_json_file
from lumina_inference.modeler import Modeler

__all__ = [
    "Modeler",
    "__version__",
    # Ingestion functions (top-level convenience)
    "build_hetero_data",
    "load_from_dict",
    "load_from_json_file",
    "load_from_json_string",
    "load_from_matpower",
    "process_json_file",
    "process_opf_dict",
]
