"""
GridSFM US Power Grid — HuggingFace Loader

Download and load power grid models and OPF results from HuggingFace Hub.
Metadata (available regions, hours, file types, state abbreviations) is
pulled from the dataset repo's ``dataset_metadata.json`` — not hardcoded —
so the loader works across dataset versions without changes.

Install:
    pip install huggingface_hub

Usage:
    from gridsfm.hf_util import GridSFM_PG_Loader

    # Basic: files stay in HuggingFace cache
    loader = GridSFM_PG_Loader("microsoft/GridSFM_US_power_grid")

    # With export_dir: the entire dataset is pre-fetched to a local directory
    # (ready for the Data Viewer, PowerModels.jl, etc.)
    loader = GridSFM_PG_Loader(
        "microsoft/GridSFM_US_power_grid",
        export_dir="./gridsfm_data",           # pre-fetches everything here
    )

    # To skip the automatic download, set pre_fetch_all=False
    loader = GridSFM_PG_Loader(
        "microsoft/GridSFM_US_power_grid",
        export_dir="./gridsfm_data",
        pre_fetch_all=False,                    # download lazily on access
    )

    # Load a single model (case-insensitive; abbreviations work too)
    model = loader.load_model("TX", hour="16h")
    model = loader.load_model("texas", hour="16h")

    # Load OPF results
    ac = loader.load_ac_results("texas", hour="16h")
    dc = loader.load_dc_results("texas", hour="16h")

    # Load everything for a region
    bundle = loader.load_bundle("texas", hour="16h")
    # bundle["model"], bundle["ac_results"], bundle["dc_results"]

    # Export a single file to a specific path
    loader.export_file("TX", "model", hour="16h", dest="./my_models/texas.json")

    # Save a (possibly modified) model dict back to JSON
    loader.save_json(model, "./my_models/texas_modified.json")

    # List available regions/states
    print(loader.list_regions())

    # Download all files to a local directory
    loader.download_all("./data")
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import dataset_info, hf_hub_download, snapshot_download

METADATA_FILENAME = "dataset_metadata.json"


class GridSFM_PG_Loader:
    """Download and load GridSFM US Power Grid data from HuggingFace Hub.

    Metadata (regions, hours, file types, abbreviations) is fetched from
    ``dataset_metadata.json`` in the dataset repo on first access and cached
    for the lifetime of the loader instance.
    """

    def __init__(self, repo_id: str, revision: Optional[str] = None,
                 cache_dir: Optional[str] = None, token: Optional[str] = None,
                 export_dir: Optional[str] = None, pre_fetch_all: bool = True):
        """
        Args:
            repo_id:       HuggingFace dataset repo (e.g. "YOUR_ORG/GridSFM_US_power_grid").
            revision:      Branch/tag/commit to use (default: main).
            cache_dir:     Local cache directory (default: HuggingFace cache).
            token:         HuggingFace token (only needed for private repos).
            export_dir:    If set, downloaded files are copied to this directory
                           with the original folder structure preserved
                           (e.g. ``export_dir/16h/texas_model.json``).  This
                           makes it easy to pass the directory to the Data
                           Viewer or other tools that expect a local data root.
            pre_fetch_all: If True (default) **and** ``export_dir`` is set, the
                           entire dataset is downloaded to ``export_dir`` during
                           initialization.  Set to False to download files lazily
                           (only when accessed via load_* / export_* methods).
        """
        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.token = token
        self.repo_type = "dataset"
        self.export_dir: Optional[Path] = Path(export_dir) if export_dir else None
        self._metadata: Optional[dict] = None

        # Verify the dataset exists on HuggingFace Hub
        try:
            dataset_info(repo_id, revision=revision, token=token)
        except Exception as exc:
            raise ValueError(
                f"Dataset '{repo_id}' is not available on HuggingFace Hub. "
                f"Check the repo ID and your network connection."
            ) from exc

        if self.export_dir is not None and pre_fetch_all:
            self.download_all(str(self.export_dir))

    # --------------------------------------------------------------------- #
    #  Metadata
    # --------------------------------------------------------------------- #

    @property
    def metadata(self) -> dict:
        """Fetch and cache dataset_metadata.json from the repo."""
        if self._metadata is None:
            self._metadata = self._load_json(METADATA_FILENAME)
            # Build lookup tables for case-insensitive + abbreviation resolution
            regions = self._metadata["regions"]
            abbrev_map: dict[str, str] = {}   # "tx" -> "texas"
            name_map: dict[str, str] = {}     # "texas" -> "texas"
            for name, info in regions.items():
                name_map[name.lower()] = name
                abbr = info.get("abbreviation")
                if abbr:
                    abbrev_map[abbr.lower()] = name
            self._metadata["_name_map"] = name_map
            self._metadata["_abbrev_map"] = abbrev_map
        return self._metadata

    @property
    def hours(self) -> list[str]:
        return self.metadata["hours"]

    @property
    def file_types(self) -> list[str]:
        return self.metadata["file_types"]

    @property
    def file_pattern(self) -> str:
        return self.metadata.get("file_pattern", "{hour}/{region}_{file_type}.json")

    # --------------------------------------------------------------------- #
    #  Core download helpers
    # --------------------------------------------------------------------- #

    def _download_file(self, filename: str) -> str:
        """Download a single file, auto-export if export_dir is set, return local path."""
        cached_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type=self.repo_type,
            revision=self.revision,
            cache_dir=self.cache_dir,
            token=self.token,
        )
        if self.export_dir is not None:
            return self._copy_to_export(cached_path, filename)
        return cached_path

    @staticmethod
    def _needs_copy(src: str, dst: Path) -> bool:
        """Return True if dst doesn't exist or differs from src."""
        if not dst.exists():
            return True
        try:
            return not dst.samefile(src)
        except (OSError, ValueError):
            return True

    def _copy_to_export(self, cached_path: str, filename: str) -> str:
        """Copy a cached file to the export directory, preserving folder structure."""
        target = self.export_dir / filename
        if self._needs_copy(cached_path, target):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_path, target)
        return str(target)

    def _load_json(self, filename: str) -> dict:
        """Download a JSON file and return its parsed contents."""
        local_path = self._download_file(filename)
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_region(self, region: str) -> str:
        """Resolve a region input (name or abbreviation, any case) to its canonical name.

        Accepts: "texas", "Texas", "TX", "tx", "new_england", etc.
        Returns the canonical lowercase name used in filenames (e.g. "texas").
        Raises ValueError if not found.
        """
        key = region.strip().lower()
        name_map = self.metadata["_name_map"]
        abbrev_map = self.metadata["_abbrev_map"]
        if key in name_map:
            return name_map[key]
        if key in abbrev_map:
            return abbrev_map[key]
        valid = sorted(name_map.keys())
        raise ValueError(
            f"Unknown region '{region}'. "
            f"Valid regions: {', '.join(valid)}"
        )

    def _validate_hour(self, hour: str) -> None:
        if hour not in self.hours:
            raise ValueError(
                f"Unknown hour '{hour}'. Valid hours: {', '.join(self.hours)}"
            )

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #

    def list_regions(self) -> list[str]:
        """Return all available region/state names."""
        return sorted(self.metadata["regions"].keys())

    def list_states(self) -> list[str]:
        """Return state names only (excludes multi-state regions)."""
        return sorted(
            name for name, info in self.metadata["regions"].items()
            if info.get("type") == "state"
        )

    def list_multi_state_regions(self) -> list[str]:
        """Return multi-state region names only."""
        return sorted(
            name for name, info in self.metadata["regions"].items()
            if info.get("type") == "region"
        )

    def list_abbreviations(self) -> dict[str, str]:
        """Return a dict mapping state abbreviations to region names."""
        return {
            info["abbreviation"]: name
            for name, info in self.metadata["regions"].items()
            if info.get("abbreviation")
        }

    def list_hours(self) -> list[str]:
        """Return available operating hours."""
        return list(self.hours)

    def list_file_types(self) -> list[str]:
        """Return available file types."""
        return list(self.file_types)

    def _make_filename(self, region: str, file_type: str, hour: str) -> str:
        """Build the repo-relative filename using the metadata pattern."""
        return self.file_pattern.format(
            hour=hour, region=region, file_type=file_type,
        )

    def load_model(self, region: str, hour: str = "16h") -> dict:
        """Load the network model JSON for a region and hour.

        Args:
            region: State or region name/abbreviation (e.g. "texas", "TX", "eastern").
                    Case-insensitive.
            hour:   Operating hour ("16h" for peak, "04h" for off-peak).

        Returns:
            Parsed JSON dict compatible with PowerModels.jl / MATPOWER.
        """
        region = self._resolve_region(region)
        self._validate_hour(hour)
        return self._load_json(self._make_filename(region, "model", hour))

    def load_ac_results(self, region: str, hour: str = "16h") -> dict:
        """Load AC-OPF results for a region and hour."""
        region = self._resolve_region(region)
        self._validate_hour(hour)
        return self._load_json(self._make_filename(region, "ac_results", hour))

    def load_dc_results(self, region: str, hour: str = "16h") -> dict:
        """Load DC-OPF results for a region and hour."""
        region = self._resolve_region(region)
        self._validate_hour(hour)
        return self._load_json(self._make_filename(region, "dc_results", hour))

    def load_bundle(self, region: str, hour: str = "16h") -> dict:
        """Load all file types for a region and hour as a single dict.

        Returns:
            e.g. {"model": {...}, "ac_results": {...}, "dc_results": {...}}
        """
        region = self._resolve_region(region)
        self._validate_hour(hour)
        return {
            ft: self._load_json(self._make_filename(region, ft, hour))
            for ft in self.file_types
        }

    def download_region(self, region: str, hour: str = "16h",
                        local_dir: Optional[str] = None) -> list[str]:
        """Download all files for a region to a local directory.

        Args:
            region:    State or region name/abbreviation (case-insensitive).
            hour:      Operating hour.
            local_dir: Target directory. Defaults to ``export_dir`` if set,
                       otherwise the current directory.

        Returns:
            List of local file paths.
        """
        region = self._resolve_region(region)
        self._validate_hour(hour)
        out = Path(local_dir) if local_dir else (self.export_dir or Path("."))
        paths = []
        for ft in self.file_types:
            filename = self._make_filename(region, ft, hour)
            cached_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                repo_type=self.repo_type,
                revision=self.revision,
                cache_dir=self.cache_dir,
                token=self.token,
            )
            target = out / filename
            if self._needs_copy(cached_path, target):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached_path, target)
            paths.append(str(target))
        return paths

    def download_all(self, local_dir: Optional[str] = None) -> str:
        """Download the entire dataset to a local directory.

        Args:
            local_dir: Target directory. Defaults to ``export_dir`` if set,
                       otherwise ``./gridsfm_pg_data``.

        Returns:
            Path to the local directory.
        """
        out = local_dir or (str(self.export_dir) if self.export_dir else "./gridsfm_pg_data")
        return snapshot_download(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            revision=self.revision,
            cache_dir=self.cache_dir,
            token=self.token,
            local_dir=out,
        )

    # --------------------------------------------------------------------- #
    #  Export / save helpers
    # --------------------------------------------------------------------- #

    def export_file(self, region: str, file_type: str, hour: str = "16h",
                    dest: Optional[str] = None) -> str:
        """Download a single file and save it to a specific path.

        Args:
            region:    State or region name/abbreviation (case-insensitive).
            file_type: One of the dataset file types (e.g. "model", "ac_results").
            hour:      Operating hour.
            dest:      Full destination file path. If None, saves into
                       ``export_dir`` (or current directory) preserving the
                       original folder structure.

        Returns:
            The path where the file was saved.
        """
        region = self._resolve_region(region)
        self._validate_hour(hour)
        valid_file_types = self.file_types
        if file_type not in valid_file_types:
            raise ValueError(
                f"Invalid file_type '{file_type}'. "
                f"Valid file types: {', '.join(valid_file_types)}"
            )
        filename = self._make_filename(region, file_type, hour)
        cached_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type=self.repo_type,
            revision=self.revision,
            cache_dir=self.cache_dir,
            token=self.token,
        )
        if dest:
            target = Path(dest)
        else:
            base = self.export_dir or Path(".")
            target = base / filename
        if self._needs_copy(cached_path, target):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_path, target)
        return str(target)

    def export_region(self, region: str, hour: str = "16h",
                      dest_dir: Optional[str] = None) -> list[str]:
        """Export all files for a region to a directory.

        Alias for ``download_region`` — provided for clarity when the intent
        is to create a working copy outside the HuggingFace cache.

        Args:
            region:   State or region name/abbreviation (case-insensitive).
            hour:     Operating hour.
            dest_dir: Target directory. Defaults to ``export_dir`` or ``"."``.

        Returns:
            List of saved file paths.
        """
        return self.download_region(region, hour, local_dir=dest_dir)

    def export_all(self, dest_dir: Optional[str] = None) -> str:
        """Export the entire dataset to a directory.

        Alias for ``download_all`` — provided for clarity.

        Args:
            dest_dir: Target directory. Defaults to ``export_dir`` or
                      ``./gridsfm_pg_data``.

        Returns:
            Path to the directory.
        """
        return self.download_all(local_dir=dest_dir)

    def save_json(self, data: dict, dest: str) -> str:
        """Save a dict (e.g. a loaded model or results) to a JSON file.

        Useful for exporting a previously loaded / modified model so it can be
        passed directly to PowerModels.jl, MATPOWER, or the Data Viewer.

        Args:
            data: The dict to serialize (model, ac_results, dc_results, etc.).
            dest: Destination file path.

        Returns:
            The path where the file was saved.
        """
        target = Path(dest)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return str(target)

    def get_export_path(self, region: str, file_type: str = "model",
                        hour: str = "16h") -> str:
        """Return the expected local path for an exported file.

        Does NOT download — just computes where the file would be saved
        given the current ``export_dir``.  Useful for constructing paths to
        pass to external tools.

        Args:
            region:    State or region name/abbreviation (case-insensitive).
            file_type: File type (e.g. "model").
            hour:      Operating hour.

        Returns:
            Expected file path as a string.

        Raises:
            ValueError: If ``export_dir`` is not set.
        """
        if self.export_dir is None:
            raise ValueError(
                "export_dir is not set. Pass export_dir= to the constructor "
                "or use export_file() / export_region() with an explicit dest."
            )
        region = self._resolve_region(region)
        self._validate_hour(hour)
        filename = self._make_filename(region, file_type, hour)
        return str(self.export_dir / filename)

    # --------------------------------------------------------------------- #
    #  Convenience: quick summary
    # --------------------------------------------------------------------- #

    @staticmethod
    def summarize_model(model: dict) -> dict:
        """Return a quick summary of a loaded model."""
        return {
            "name": model.get("name", "unknown"),
            "n_buses": len(model.get("bus", {})),
            "n_branches": len(model.get("branch", {})),
            "n_generators": len(model.get("gen", {})),
            "n_loads": len(model.get("load", {})),
            "n_shunts": len(model.get("shunt", {})),
            "n_dclines": len(model.get("dcline", {})),
            "total_load_mw": sum(
                l["pd"] for l in model.get("load", {}).values()
            ) * model.get("baseMVA", 100.0),
            "target_datetime": model.get("target_datetime"),
            "balancing_authority": model.get("balancing_authority"),
            "is_multi_state": model.get("is_multi_state", False),
        }


# --------------------------------------------------------------------- #
#  CLI demo
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and inspect GridSFM US Power Grid models from HuggingFace."
    )
    parser.add_argument(
        "repo_id",
        help="HuggingFace dataset repo ID (e.g. 'YOUR_ORG/GridSFM_US_power_grid')",
    )
    parser.add_argument(
        "--region", default="texas",
        help="Region or state to load (default: texas)",
    )
    parser.add_argument(
        "--hour", default="16h",
        help="Operating hour (default: 16h)",
    )
    parser.add_argument(
        "--download-all", action="store_true",
        help="Download the entire dataset",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_regions",
        help="List all available regions and exit",
    )
    parser.add_argument(
        "--export-dir", default=None,
        help="Download the entire dataset to this directory on init. "
             "Use --no-pre-fetch to skip the automatic download.",
    )
    parser.add_argument(
        "--no-pre-fetch", action="store_true",
        help="With --export-dir, do not pre-fetch the entire dataset on init.",
    )
    args = parser.parse_args()

    loader = GridSFM_PG_Loader(
        args.repo_id,
        export_dir=args.export_dir,
        pre_fetch_all=not args.no_pre_fetch,
    )

    if args.list_regions:
        print("Available regions/states:")
        for name in loader.list_regions():
            info = loader.metadata["regions"][name]
            abbr = info.get("abbreviation") or ""
            rtype = info.get("type", "")
            print(f"  {name:20s} {abbr:>2s}  ({rtype})")
        print(f"\nAvailable hours: {', '.join(loader.list_hours())}")
        print(f"File types: {', '.join(loader.list_file_types())}")
    elif args.download_all:
        out = args.export_dir or "./gridsfm_pg_data"
        print(f"Downloading entire dataset to {out}...")
        path = loader.download_all(out)
        print(f"Done. Files saved to: {path}")
    else:
        print(f"Loading {args.region} ({args.hour})...")
        model = loader.load_model(args.region, args.hour)
        summary = loader.summarize_model(model)
        for k, v in summary.items():
            print(f"  {k}: {v}")
