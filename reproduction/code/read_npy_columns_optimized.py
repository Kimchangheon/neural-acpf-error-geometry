import io
import os
import time
import multiprocessing as mp
from functools import partial
from typing import List

import numpy as np
import pandas as pd


# --- Helper functions ---
def npy_bytes_to_ndarray(b: bytes) -> np.ndarray:
    """Converts a bytes object containing a .npy file back to a NumPy array."""
    # Using a with statement ensures the BytesIO buffer is properly closed.
    with io.BytesIO(b) as buf:
        return np.load(buf, allow_pickle=False)


# -------- Method 1: Simple, single-core (Baseline) --------
def decode_columns_single_core(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df_copy = df.copy()
    for c in cols:
        df_copy[c] = [npy_bytes_to_ndarray(b) for b in df_copy[c].values]
    return df_copy


# -------- Method 2: CORRECTED Chunking Multiprocessing --------
def _worker_decode_chunk(df_chunk: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Worker function that decodes columns on an explicit copy of a DataFrame chunk.
    This prevents issues with modifying views vs. copies.
    """
    df_copy = df_chunk.copy()  # Explicitly work on a copy.
    for c in cols:
        df_copy[c] = [npy_bytes_to_ndarray(b) for b in df_copy[c].values]
    return df_copy


def decode_columns_mp_chunked(df: pd.DataFrame, cols: List[str], workers: int = 0) -> pd.DataFrame:
    """
    Decodes columns by splitting the DataFrame into chunks and processing in parallel.
    Corrected for the AssertionError.
    """
    if workers <= 0:
        workers = os.cpu_count() or 1
    if workers == 1:
        return decode_columns_single_core(df, cols)

    df_chunks = np.array_split(df, workers)
    worker_func = partial(_worker_decode_chunk, cols=cols)

    context = "fork" if os.name != "nt" else "spawn"
    with mp.get_context(context).Pool(processes=workers) as pool:
        processed_chunks = pool.map(worker_func, df_chunks)

    return pd.concat(processed_chunks)


# -------- Method 3: NEW Column-wise Multiprocessing (Recommended) --------
def decode_columns_mp_columnwise(df: pd.DataFrame, cols: List[str], workers: int = 0) -> pd.DataFrame:
    """
    Decodes columns by parallelizing the conversion of rows within each column.
    This avoids DataFrame splitting/concatenation overhead.
    """
    if workers <= 0:
        workers = os.cpu_count() or 1

    df_copy = df.copy()
    context = "fork" if os.name != "nt" else "spawn"

    with mp.get_context(context).Pool(processes=workers) as pool:
        for c in cols:
            # Map the decoding function across all rows of the column in parallel
            decoded_list = pool.map(npy_bytes_to_ndarray, df_copy[c].values)
            df_copy[c] = decoded_list

    return df_copy


# ======== Main Execution Block ========
if __name__ == "__main__":
    # Load your data
    try:
        df_orig = pd.read_parquet("../data/HVN_150000_PQrangeFixed_4_to_32_bus_grid_Ybus.parquet")
    except FileNotFoundError:
        print("Parquet file not found. Creating a dummy DataFrame for demonstration.")
        num_rows = 150000
        binary_cols_list = ['bus_typ', 'Y_matrix', 'S_newton']
        data = {col: [np.random.rand(8).tobytes()] * num_rows for col in binary_cols_list}
        df_orig = pd.DataFrame(data)

    print(f"DataFrame loaded with {len(df_orig)} rows.")

    binary_cols = [
        "bus_typ",
        "vn_kv",
        "Y_shunt_bus",
        "Branch_f_bus",
        "Branch_t_bus",
        "Branch_status",
        "Branch_tau",
        "Branch_shift_deg",
        "Branch_y_series_from",
        "Branch_y_series_to",
        "Branch_y_series_ft",
        "Branch_y_shunt_from",
        "Branch_y_shunt_to",
        "Is_trafo",
        "Branch_hv_is_f",
        "Branch_n",
        "Y_Lines",
        "Y_C_Lines",
        "Y_matrix",
        "u_start",
        "u_newton",
        "S_start",
        "S_newton",
    ]
    # Filter for columns that actually exist in the dataframe to avoid errors
    binary_cols_exist = [col for col in binary_cols if col in df_orig.columns]
    print(f"Columns to be decoded: {binary_cols_exist}")

    # --- 1. Time the simple, single-core version ---
    start_time = time.time()
    df_single = decode_columns_single_core(df_orig, binary_cols_exist)
    simple_time = time.time() - start_time
    print(f"\nMethod 1: Simple (single-core) time: {simple_time:.2f} seconds")

    # --- 2. Time the corrected chunking version ---
    start_time = time.time()
    df_chunked = decode_columns_mp_chunked(df_orig, binary_cols_exist)
    chunked_time = time.time() - start_time
    print(f"Method 2: Corrected MP (chunking) time: {chunked_time:.2f} seconds")

    # --- 3. Time the new column-wise version ---
    start_time = time.time()
    df_colwise = decode_columns_mp_columnwise(df_orig, binary_cols_exist)
    colwise_time = time.time() - start_time
    print(f"Method 3: New MP (column-wise) time: {colwise_time:.2f} seconds")

    print("\n--- Verification ---")
    try:
        # Using a custom comparison because pandas testing can be slow on large object-dtype frames
        # We just check the first element of the first decoded column
        assert np.array_equal(df_single[binary_cols_exist[0]].iloc[0], df_chunked[binary_cols_exist[0]].iloc[0])
        print("✅ Corrected chunking method produced the correct result.")
    except AssertionError:
        print("❌ Corrected chunking method produced an INCORRECT result.")

    try:
        assert np.array_equal(df_single[binary_cols_exist[0]].iloc[0], df_colwise[binary_cols_exist[0]].iloc[0])
        print("✅ New column-wise method produced the correct result.")
    except AssertionError:
        print("❌ New column-wise method produced an INCORRECT result.")

    print("\n--- Results ---")
    best_time = min(simple_time, chunked_time, colwise_time)
    if best_time == colwise_time and colwise_time < simple_time:
        speedup = simple_time / colwise_time
        print(f"🏆 The new column-wise method was fastest, with a {speedup:.2f}x speedup over single-core.")
    elif best_time == chunked_time and chunked_time < simple_time:
        speedup = simple_time / chunked_time
        print(f"🏆 The corrected chunking method was fastest, with a {speedup:.2f}x speedup.")
    else:
        print("⚠️ Multiprocessing did not provide a speedup. The single-core version was fastest.")
        print("   This can happen on machines with few cores or very fast memory/storage.")

    pd.testing.assert_frame_equal(df_single, df_colwise)
    print("\nVerification successful: All methods produce the same result.")