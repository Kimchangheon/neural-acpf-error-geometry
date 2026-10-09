"""Dataset utilities for LUMINA inference.

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

from lumina_inference.dataset.ingestion import (
    build_hetero_data,
    load_from_dict,
    load_from_json_file,
    load_from_json_string,
    load_from_matpower,
    process_opf_dict,
)
from lumina_inference.dataset.opf_dataset import OPFDataset, process_json_file
from lumina_inference.dataset.validation import (
    FeatureDimensionError,
    LuminaIngestionError,
    MissingDataError,
    SchemaValidationError,
    detect_schema,
    validate_hetero_data,
    validate_opf_schema,
)

__all__ = [
    # Dataset classes
    "OPFDataset",
    # Ingestion functions
    "build_hetero_data",
    "load_from_dict",
    "load_from_json_file",
    "load_from_json_string",
    "load_from_matpower",
    "process_json_file",
    "process_opf_dict",
    # Validation
    "detect_schema",
    "validate_hetero_data",
    "validate_opf_schema",
    # Exceptions
    "FeatureDimensionError",
    "LuminaIngestionError",
    "MissingDataError",
    "SchemaValidationError",
]
