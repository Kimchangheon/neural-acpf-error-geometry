#!/usr/bin/env python3
"""
Minimal data viewer server for the power grid data release.

Serves model JSONs and OPF results from the release directory structure:
    <data_dir>/16h/<state>_model.json
    <data_dir>/16h/<state>_dc_results.json
    <data_dir>/16h/<state>_ac_results.json
    <data_dir>/16h/<state>_interfaces.json
    <data_dir>/04h/...

Usage:
    python serve.py [--data-dir ../] [--port 8050]
"""

import argparse
import json
import os
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn


class Handler(SimpleHTTPRequestHandler):
    """HTTP handler that serves static files + discovery APIs."""

    data_dir = Path(".")

    def do_GET(self):
        path = self.path.split("?")[0]

        # API: list available states and regions
        if path == "/api/states":
            return self._json_response(self._discover_datasets())

        # API: list available hours
        if path == "/api/hours":
            hours = []
            for d in sorted(self.data_dir.iterdir()):
                if d.is_dir() and re.match(r"\d+h$", d.name):
                    h = int(d.name.replace("h", ""))
                    label = f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"
                    hours.append({"value": d.name, "label": label, "hour": h})
            return self._json_response(hours)

        # Data files: /data/<hour>/<filename>
        if path.startswith("/data/"):
            rel = path[len("/data/"):]
            file_path = (self.data_dir / rel).resolve()
            if not file_path.is_relative_to(self.data_dir.resolve()):
                self.send_error(403, "Forbidden")
                return
            if file_path.is_file() and file_path.suffix == ".json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            self.send_error(404, f"Not found: {rel}")
            return

        # Static files (index.html, etc.) from script directory
        return super().do_GET()

    def _discover_datasets(self):
        """Find all unique state/region names across hour directories."""
        names = set()
        for hour_dir in self.data_dir.iterdir():
            if not hour_dir.is_dir() or not re.match(r"\d+h$", hour_dir.name):
                continue
            for f in hour_dir.iterdir():
                if f.name.endswith("_model.json"):
                    name = f.name.replace("_model.json", "")
                    names.add(name)
        datasets = []
        for name in sorted(names):
            label = name.replace("_", " ").title()
            # Detect multi-state regions (no "power_grid_bbox" and known region names)
            is_region = name in {
                "new_england", "pacific_nw", "desert_sw",
                "western", "eastern", "pjm", "ercot",
                "miso", "spp", "continental_us",
            }
            datasets.append({
                "id": name,
                "label": label,
                "is_region": is_region,
            })
        return datasets

    def _json_response(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Quieter logging
        if args and "404" not in str(args[1] if len(args) > 1 else ""):
            return
        super().log_message(fmt, *args)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description="Power Grid Data Viewer")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent.parent,
                        help="Root of data release (contains 16h/, 04h/)")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    # Resolve data directory
    data_dir = args.data_dir.resolve()
    if not any(d.is_dir() and re.match(r"\d+h$", d.name) for d in data_dir.iterdir()):
        print(f"⚠️  No hour directories (16h/, 04h/) found in {data_dir}")
        print(f"   Expected structure: {data_dir}/16h/<state>_model.json")
        return

    # Serve static files from script directory, data from data_dir
    Handler.data_dir = data_dir
    os.chdir(Path(__file__).parent)

    server = ThreadedHTTPServer(("", args.port), Handler)
    print(f"🗺️  Power Grid Data Viewer")
    print(f"   Data: {data_dir}")
    print(f"   URL:  http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
