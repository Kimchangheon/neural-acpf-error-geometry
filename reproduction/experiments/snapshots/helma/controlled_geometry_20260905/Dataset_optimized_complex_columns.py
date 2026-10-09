import io
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from read_npy_columns_optimized import decode_columns_mp_columnwise, npy_bytes_to_ndarray

FLOAT_DTYPE = np.float32
COMPLEX_DTYPE = np.complex64


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _as_np(x, dtype=None):
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _all_zero_complex_fast(cell) -> bool:
    a = np.asarray(cell)
    if a.size == 0:
        return True
    if np.iscomplexobj(a):
        return bool(np.all((a.real == 0) & (a.imag == 0)))
    return bool(np.all(a == 0))


def reconstruct_Y_pandapower_branchrows_direct_SI_np(
    N: int,
    Branch_f_bus: np.ndarray,
    Branch_t_bus: np.ndarray,
    Branch_status: np.ndarray,
    Branch_tau: np.ndarray,
    Branch_shift_deg: np.ndarray,
    Branch_y_series_from: np.ndarray,
    Branch_y_series_to: np.ndarray,
    Branch_y_series_ft: np.ndarray,
    Branch_y_shunt_from: np.ndarray,
    Branch_y_shunt_to: np.ndarray,
    Y_shunt_bus: np.ndarray,
    Vbase_bus: Optional[np.ndarray] = None,
    out_dtype=COMPLEX_DTYPE,
) -> np.ndarray:
    """
    Reconstruct dense Ybus in SI from one metadata row per PPC branch row.
    """
    Y = np.zeros((N, N), dtype=np.complex128)
    Y[np.diag_indices(N)] += Y_shunt_bus.astype(np.complex128)

    nl = len(Branch_f_bus)
    for k in range(nl):
        if int(Branch_status[k]) == 0:
            continue

        f = int(Branch_f_bus[k])
        t = int(Branch_t_bus[k])

        tau = float(Branch_tau[k])
        theta = np.deg2rad(float(Branch_shift_deg[k]))
        a = tau * np.exp(1j * theta)

        y_from = complex(Branch_y_series_from[k])
        y_to   = complex(Branch_y_series_to[k])
        y_ft   = complex(Branch_y_series_ft[k])
        y_tf   = y_ft
        ysh_f  = complex(Branch_y_shunt_from[k])
        ysh_t  = complex(Branch_y_shunt_to[k])

        if Vbase_bus is not None:
            Vf = float(Vbase_bus[f])
            Vt = float(Vbase_bus[t])
            if Vf > 0.0 and Vt > 0.0:
                y_ft = y_from * (Vf / Vt)
                y_tf = y_to * (Vt / Vf)

        Yff = (y_from + ysh_f / 2.0) / (a * np.conj(a))
        Ytt = (y_to   + ysh_t / 2.0)
        Yft = -y_ft / np.conj(a)
        Ytf = -y_tf / a

        Y[f, f] += Yff
        Y[t, t] += Ytt
        Y[f, t] += Yft
        Y[t, f] += Ytf

    return Y.astype(out_dtype, copy=False)


def _zero_q_at_slack_pv(S: np.ndarray, bus_typ: np.ndarray) -> np.ndarray:
    """Zero Im(S) at slack (type 1) and PV (type 2) buses; Q is not specified there."""
    mask = (np.asarray(bus_typ) == 1) | (np.asarray(bus_typ) == 2)
    if not mask.any():
        return S
    out = np.array(S, copy=True)
    out[mask] = out[mask].real.astype(out.dtype)
    return out


def ybus_si_to_pu(Y_si: np.ndarray, Vbase_bus: np.ndarray, S_base: float, out_dtype=COMPLEX_DTYPE) -> np.ndarray:
    """
    Entrywise Ybus SI -> pu with local bus bases:
        Y_pu[i,j] = Y_SI[i,j] * V_i * V_j / S_base
    """
    scale = np.outer(Vbase_bus, Vbase_bus) / float(S_base)
    return (Y_si * scale).astype(out_dtype, copy=False)


def ysh_bus_si_to_pu(Ysh_si: np.ndarray, Vbase_bus: np.ndarray, S_base: float, out_dtype=COMPLEX_DTYPE) -> np.ndarray:
    """
    Bus shunt SI -> pu:
        Ysh_pu[i] = Ysh_SI[i] * V_i^2 / S_base
    """
    scale = (Vbase_bus ** 2) / float(S_base)
    return (Ysh_si * scale).astype(out_dtype, copy=False)


def u_si_to_pu_per_bus(u_si: np.ndarray, Vbase_bus: np.ndarray, out_dtype=COMPLEX_DTYPE) -> np.ndarray:
    """
    Voltage SI -> pu per bus:
        U_pu[i] = U_SI[i] / V_i
    """
    return (u_si / Vbase_bus).astype(out_dtype, copy=False)


def s_si_to_pu(S_si: np.ndarray, S_base: float, out_dtype=COMPLEX_DTYPE) -> np.ndarray:
    """
    Power SI -> pu:
        S_pu[i] = S_SI[i] / S_base
    """
    return (S_si / float(S_base)).astype(out_dtype, copy=False)


# --------------------------------------------------------------------------- #
# main Dataset
# --------------------------------------------------------------------------- #

class ChanghunDataset(Dataset):
    """
    Dataset for the branch-row parquet schema with CORRECT multi-voltage per-unit conversion.

    Expected per-row parquet fields:
      bus_number, branch_number, gridtype, U_base, S_base
      bus_typ, vn_kv, Y_shunt_bus
      Branch_f_bus, Branch_t_bus, Branch_status, Branch_tau, Branch_shift_deg
      Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft
      Branch_y_shunt_from, Branch_y_shunt_to
      Is_trafo, Branch_hv_is_f, Branch_n
      Y_Lines, Y_C_Lines
      u_start, u_newton, S_start, S_newton
      optional: Y_matrix, u_nr_trajectory, nr_misinf_history,
                nr_step_history, nr_trajectory_length

    If per_unit=True:
      - Ybus uses local (Vi,Vj) base conversion
      - branch-row admittances use correct local from/to bases
      - voltages use per-bus base
      - powers use S_base
    """
    __slots__ = (
        "rows",
        "_per_unit",
        "_device",
        "_no_cache_dense_ybus",
        "_lazy_row_groups",
        "_row_group_cache_size",
        "_binary_cols_master",
        "_read_columns",
        "_parquet_files",
        "_row_group_metas",
        "_row_group_ends",
        "_row_group_cache",
        "_num_rows",
        "_target_S_base",
        "_share_ybus",
        "_shared_ybus_tensor",
        "_share_grid",
        "_shared_grid_tensors",
        "_shared_grid_numpy",
        "_complex_np_dtype",
        "_complex_torch_dtype",
        "_real_np_dtype",
        "_real_torch_dtype",
    )

    def __init__(
        self,
        paths: Union[str, Path, List[str], List[Path]],
        *,
        per_unit: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        no_cache_dense_ybus: bool = False,
        lazy_row_groups: bool = False,
        row_group_cache_size: int = 2,
        target_S_base: Optional[float] = None,
        share_ybus: bool = False,
        share_grid: bool = False,
        complex_dtype: str = "complex64",
        extra_binary_columns: Optional[List[str]] = None,
        mask_q_at_slack_pv: bool = False,
    ):
        self._per_unit = bool(per_unit)
        # Opt-in backstop. Reactive power at slack/PV buses is not a specified
        # quantity -- the solver fixes |V| there and solves for Q -- so whatever
        # sits in that slot of S_start is arbitrary. Some CGMES exports park a
        # reactive-capability figure there (LVN heo1: Q = -2507 pu at one 380 kV
        # PV bus vs a true maximum of 8.4 pu), which corrupts any consumer that
        # reads S_start without masking by bus type. Default False preserves the
        # existing behaviour; set True to zero those entries at load time.
        self._mask_q_at_slack_pv = bool(mask_q_at_slack_pv)
        self._device = torch.device(device) if device is not None else None
        self._no_cache_dense_ybus = bool(no_cache_dense_ybus)
        self._lazy_row_groups = bool(lazy_row_groups)
        self._row_group_cache_size = max(1, int(row_group_cache_size))
        self._parquet_files = []
        self._row_group_metas = []
        self._row_group_ends = []
        self._row_group_cache = OrderedDict()
        self._num_rows = 0
        complex_dtype = str(complex_dtype).lower()
        if complex_dtype in ("complex128", "complex_double", "cdouble", "128"):
            self._complex_np_dtype = np.complex128
            self._complex_torch_dtype = torch.complex128
            self._real_np_dtype = np.float64
            self._real_torch_dtype = torch.float64
        elif complex_dtype in ("complex64", "complex_float", "cfloat", "64"):
            self._complex_np_dtype = np.complex64
            self._complex_torch_dtype = torch.complex64
            self._real_np_dtype = np.float32
            self._real_torch_dtype = torch.float32
        else:
            raise ValueError(
                f"Unsupported complex_dtype={complex_dtype!r}; use 'complex64' or 'complex128'."
            )
        # Optional in-memory rebase of S_base for per-unit conversion.
        # Default (None) keeps the parquet's S_base as-is. Pass an explicit
        # value (e.g. target_S_base=100e6) to normalize the per-unit scale
        # across datasets. Useful for distribution-network parquets shipped
        # with S_base=1 MVA (e.g. LVN Heo1, pandapower default) whose per-unit
        # S_pu / Y_pu are ~100x larger than IEEE/PEGASE benchmarks, causing
        # Armijo to reject every step and pure-PINN to diverge. Mathematically
        # equivalent to a base change (V_pu unchanged; Y_pu and S_pu both
        # scale by S_raw / S_target); the power-flow equation is preserved.
        # Recommended long-term fix: rebase the parquet file itself.
        self._target_S_base = (
            float(target_S_base) if target_S_base is not None else None
        )
        # share_ybus: when True, assume every row in the dataset shares the
        # SAME admittance matrix (e.g. a single grid sampled under different
        # load/generation perturbations -- which is exactly how the LVN /
        # case300 perturbation datasets are generated). The Ybus is then
        # computed from row 0 once, stored as self._shared_ybus_tensor, and
        # re-used for every __getitem__ -- skipping per-row dense Y
        # reconstruction (~722^2 complex = 4MB allocation + Python branch
        # loop) and the parquet read of Branch_y_* / Y_shunt_bus / Y_matrix
        # columns. For LVN 36k this removes the single biggest source of
        # lazy-mode disk thrashing.
        self._share_ybus = bool(share_ybus)
        self._shared_ybus_tensor: Optional[torch.Tensor] = None
        # share_grid: when True, all grid-only tensors (topology, branch
        # admittances, vn_kv, bus_type, Y_shunt_bus, Y_Lines, Y_C_Lines,
        # transformer flags) are built once from row 0 and re-used for every
        # __getitem__. Implies share_ybus=True. Only per-row-varying tensors
        # (u_newton, S_start, S_newton, U_newton, V_newton) are rebuilt per
        # call. Safe for the same assumption as share_ybus: every row
        # describes the SAME grid sampled under different perturbations.
        # NOTE: u_start is included in the shared set only when it is also
        # constant across rows (e.g. flat-start parquets). DC-informed
        # parquets have row-dependent u_start so share_grid will detect the
        # mismatch on the first comparison and fall back to per-row build
        # for that field.
        self._share_grid = bool(share_grid)
        # Field-name -> torch.Tensor for direct row dict insertion.
        self._shared_grid_tensors: Dict[str, torch.Tensor] = {}
        # Small numpy arrays needed during per-row per-unit math
        # (Vbase_bus for u_* division). Populated alongside grid tensors.
        self._shared_grid_numpy: Dict[str, np.ndarray] = {}
        if self._share_grid:
            # share_grid implies share_ybus
            self._share_ybus = True

        if isinstance(paths, (str, Path)):
            paths = [paths]

        self._binary_cols_master = [
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
            "u_start",
            "u_newton",
            "S_start",
            "S_newton",
            "u_nr_trajectory",
            "nr_misinf_history",
            "nr_step_history",
        ]
        if not self._no_cache_dense_ybus:
            self._binary_cols_master.insert(self._binary_cols_master.index("u_start"), "Y_matrix")

        # Opt-in passthrough for per-scenario columns that the power-flow
        # datasets do not carry (the AC-OPF decision space: per-bus generation
        # limits, available power, cost slope, voltage band, branch ratings).
        # They are decoded and handed through untouched: no per-unit math and
        # no share_grid caching, since they vary per scenario.
        self._extra_binary_columns = [str(c) for c in (extra_binary_columns or [])]
        self._binary_cols_master = self._binary_cols_master + [
            c for c in self._extra_binary_columns if c not in self._binary_cols_master
        ]

        self._read_columns = [
            "bus_number",
            "branch_number",
            "S_base",
            "U_base",
            *self._binary_cols_master,
            "nr_trajectory_length",
            "nr_trajectory_converged",
            "nr_trajectory_final_misinf",
            "nr_trajectory_final_vs_label_max_pu",
        ]

        if self._lazy_row_groups:
            self.rows = None
            self._build_lazy_row_group_index(paths)
        else:
            self.rows = []
            self._load_eager_rows(paths)

    def _present_columns(self, schema_names) -> List[str]:
        schema_names = set(schema_names)
        return [col for col in self._read_columns if col in schema_names]

    def _attach_extra_columns(self, row: Dict[str, Any], r, available_columns) -> Dict[str, Any]:
        """Decode opt-in extra columns as plain tensors, in stored units."""
        if not self._extra_binary_columns:
            return row
        for col in self._extra_binary_columns:
            if col not in available_columns:
                continue
            value = r[col] if not hasattr(r, "get") else r.get(col, None)
            if value is None:
                continue
            arr = _as_np(value)
            if np.iscomplexobj(arr):
                tensor_dtype = self._complex_torch_dtype
                arr = arr.astype(self._complex_np_dtype, copy=False)
            elif np.issubdtype(arr.dtype, np.integer):
                tensor_dtype = torch.int64
                arr = arr.astype(np.int64, copy=False)
            else:
                tensor_dtype = self._real_torch_dtype
                arr = arr.astype(self._real_np_dtype, copy=False)
            row[col] = torch.as_tensor(arr, dtype=tensor_dtype, device=self._device)
        return row

    def _attach_nr_trajectory(
        self,
        row: Dict[str, Any],
        r,
        available_columns,
        Vbase_bus: np.ndarray,
        N: int,
    ) -> Dict[str, Any]:
        """Decode an optional variable-length Newton path in stored SI units."""
        available_columns = set(available_columns)
        if "u_nr_trajectory" not in available_columns:
            return row

        trajectory = _as_np(r["u_nr_trajectory"], self._complex_np_dtype)
        trajectory = trajectory.reshape(-1, N)
        stored_length = int(r["nr_trajectory_length"])
        if trajectory.shape[0] != stored_length:
            raise ValueError(
                "u_nr_trajectory length disagrees with nr_trajectory_length: "
                f"{trajectory.shape[0]} != {stored_length}"
            )
        if stored_length < 2:
            raise ValueError(
                "a trainable Newton trajectory must contain at least the "
                f"initial and next states; got {stored_length}"
            )
        if self._per_unit:
            trajectory = (
                trajectory / np.asarray(Vbase_bus, dtype=np.float64).reshape(1, -1)
            ).astype(self._complex_np_dtype, copy=False)

        polar = np.stack(
            [
                np.abs(trajectory).astype(self._real_np_dtype),
                np.angle(trajectory).astype(self._real_np_dtype),
            ],
            axis=-1,
        )
        row["U_nr_trajectory"] = torch.as_tensor(
            trajectory, dtype=self._complex_torch_dtype, device=self._device
        )
        row["V_nr_trajectory"] = torch.as_tensor(
            polar, dtype=self._real_torch_dtype, device=self._device
        )
        row["nr_trajectory_length"] = torch.tensor(
            stored_length, dtype=torch.long, device=self._device
        )

        for source, target in (
            ("nr_misinf_history", "nr_misinf_history"),
            ("nr_step_history", "nr_step_history"),
        ):
            if source in available_columns:
                row[target] = torch.as_tensor(
                    _as_np(r[source], self._real_np_dtype),
                    dtype=self._real_torch_dtype,
                    device=self._device,
                )
        for source, dtype in (
            ("nr_trajectory_converged", torch.int8),
            ("nr_trajectory_final_misinf", self._real_torch_dtype),
            ("nr_trajectory_final_vs_label_max_pu", self._real_torch_dtype),
        ):
            if source in available_columns:
                row[source] = torch.tensor(r[source], dtype=dtype, device=self._device)
        return row

    def _build_row_dict(self, r: Union[pd.Series, Dict[str, Any]], available_columns) -> Dict[str, Any]:
        to_t = lambda x, dtype=None: torch.as_tensor(x, dtype=dtype, device=self._device)
        available_columns = set(available_columns)

        N = int(r["bus_number"])
        nl = int(r["branch_number"])

        S_base_raw = float(r["S_base"])
        U_base_scalar = float(r["U_base"])

        # ---- temporary S_base rebase (per-unit scale normalization) ----
        # See __init__ for rationale. Only active when per_unit=True and a
        # target_S_base was set. Mathematically equivalent to a base change:
        # V_pu unchanged, Y_pu and S_pu scale by (S_raw / S_target), and the
        # power-flow equation S = V * conj(Y V) is preserved (both sides scale
        # by the same factor). The model sees identical physics but on a
        # numerical scale matching the IEEE/PEGASE training regime.
        if self._per_unit and self._target_S_base is not None:
            S_base = self._target_S_base
        else:
            S_base = S_base_raw

        # =================================================================
        # share_grid FAST PATH: cache is already populated for the grid.
        # Skip every grid-related parquet decode, per-unit math, dense Y
        # reconstruction, and tensor allocation. Only read the
        # perturbation-dependent columns (u_*, S_*), do their per-unit math
        # using the cached Vbase_bus, and splice the cached grid tensors
        # into the returned row dict.
        # =================================================================
        if self._share_grid and self._shared_grid_tensors:
            Vbase_bus_cached = self._shared_grid_numpy["Vbase_bus"]
            u_start_p = _as_np(r["u_start"], self._complex_np_dtype).reshape(N)
            u_newton_p = _as_np(r["u_newton"], self._complex_np_dtype).reshape(N)
            S_start_p = _as_np(r["S_start"], self._complex_np_dtype).reshape(N)
            if self._mask_q_at_slack_pv:
                S_start_p = _zero_q_at_slack_pv(
                    S_start_p, self._shared_grid_numpy["bus_typ"]
                )
            S_newton_p = _as_np(r["S_newton"], self._complex_np_dtype).reshape(N)
            if self._per_unit:
                u_start_p = u_si_to_pu_per_bus(u_start_p, Vbase_bus_cached, self._complex_np_dtype)
                u_newton_p = u_si_to_pu_per_bus(u_newton_p, Vbase_bus_cached, self._complex_np_dtype)
                S_start_p = s_si_to_pu(S_start_p, S_base, self._complex_np_dtype)
                S_newton_p = s_si_to_pu(S_newton_p, S_base, self._complex_np_dtype)
            V_start_p = np.stack(
                [np.abs(u_start_p).astype(self._real_np_dtype),
                 np.angle(u_start_p).astype(self._real_np_dtype)],
                axis=1,
            )
            V_newton_p = np.stack(
                [np.abs(u_newton_p).astype(self._real_np_dtype),
                 np.angle(u_newton_p).astype(self._real_np_dtype)],
                axis=1,
            )
            row: Dict[str, Any] = {
                "N": N,
                "nl": nl,
                "S_base": to_t(np.array(S_base, dtype=self._real_np_dtype), dtype=self._real_torch_dtype),
                "U_base": to_t(np.array(U_base_scalar, dtype=self._real_np_dtype), dtype=self._real_torch_dtype),
                "U_start": to_t(u_start_p, dtype=self._complex_torch_dtype),
                "U_newton": to_t(u_newton_p, dtype=self._complex_torch_dtype),
                "V_start": to_t(V_start_p, dtype=self._real_torch_dtype),
                "V_newton": to_t(V_newton_p, dtype=self._real_torch_dtype),
                "S_start": to_t(S_start_p, dtype=self._complex_torch_dtype),
                "S_newton": to_t(S_newton_p, dtype=self._complex_torch_dtype),
            }
            row.update(self._shared_grid_tensors)
            row = self._attach_nr_trajectory(
                row, r, available_columns, Vbase_bus_cached, N
            )
            return self._attach_extra_columns(row, r, available_columns)

        bus_typ = _as_np(r["bus_typ"], np.int64).reshape(N)
        vn_kv = _as_np(r["vn_kv"], self._real_np_dtype).reshape(N)
        Vbase_bus = (vn_kv.astype(np.float64) * 1e3).astype(np.float64)

        Y_shunt_bus = _as_np(r["Y_shunt_bus"], self._complex_np_dtype).reshape(N)

        Branch_f_bus = _as_np(r["Branch_f_bus"], np.int64).reshape(nl)
        Branch_t_bus = _as_np(r["Branch_t_bus"], np.int64).reshape(nl)
        Branch_status = _as_np(r["Branch_status"], np.int8).reshape(nl)
        Branch_tau = _as_np(r["Branch_tau"], self._real_np_dtype).reshape(nl)
        Branch_shift_deg = _as_np(r["Branch_shift_deg"], self._real_np_dtype).reshape(nl)

        Branch_y_series_from = _as_np(r["Branch_y_series_from"], self._complex_np_dtype).reshape(nl)
        Branch_y_series_to = _as_np(r["Branch_y_series_to"], self._complex_np_dtype).reshape(nl)
        Branch_y_series_ft = _as_np(r["Branch_y_series_ft"], self._complex_np_dtype).reshape(nl)

        Branch_y_shunt_from = _as_np(r["Branch_y_shunt_from"], self._complex_np_dtype).reshape(nl)
        Branch_y_shunt_to = _as_np(r["Branch_y_shunt_to"], self._complex_np_dtype).reshape(nl)

        Is_trafo = _as_np(r["Is_trafo"], np.int8).reshape(nl)
        Branch_hv_is_f = _as_np(r["Branch_hv_is_f"], np.int8).reshape(nl)
        Branch_n = _as_np(r["Branch_n"], self._real_np_dtype).reshape(nl)

        Y_Lines = _as_np(r["Y_Lines"], self._complex_np_dtype).reshape(nl)
        Y_C_Lines = _as_np(r["Y_C_Lines"], self._real_np_dtype).reshape(nl)

        u_start = _as_np(r["u_start"], self._complex_np_dtype).reshape(N)
        u_newton = _as_np(r["u_newton"], self._complex_np_dtype).reshape(N)
        S_start = _as_np(r["S_start"], self._complex_np_dtype).reshape(N)
        S_newton = _as_np(r["S_newton"], self._complex_np_dtype).reshape(N)
        if self._mask_q_at_slack_pv:
            S_start = _zero_q_at_slack_pv(S_start, bus_typ)

        Ybus = None
        # Fast path: share_ybus=True and we've already built the shared tensor.
        # Skip all dense Y reconstruction work and re-use the cached tensor.
        # Caller-visible behaviour is identical because Y is grid-only data
        # and every row in this dataset shares the same grid.
        reuse_shared_ybus = (
            self._share_ybus
            and not self._no_cache_dense_ybus
            and self._shared_ybus_tensor is not None
        )
        if not self._no_cache_dense_ybus and not reuse_shared_ybus:
            if "Y_matrix" in available_columns and r.get("Y_matrix", None) is not None:
                Ybus_si = _as_np(r["Y_matrix"], self._complex_np_dtype).reshape(N, N)
            else:
                Ybus_si = reconstruct_Y_pandapower_branchrows_direct_SI_np(
                    N,
                    Branch_f_bus, Branch_t_bus, Branch_status,
                    Branch_tau, Branch_shift_deg,
                    Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
                    Branch_y_shunt_from, Branch_y_shunt_to,
                    Y_shunt_bus,
                    Vbase_bus=Vbase_bus,
                    out_dtype=self._complex_np_dtype,
                )

        if self._per_unit:
            if not self._no_cache_dense_ybus and not reuse_shared_ybus:
                Ybus = ybus_si_to_pu(Ybus_si, Vbase_bus, S_base, self._complex_np_dtype)

            Y_shunt_bus = ysh_bus_si_to_pu(Y_shunt_bus, Vbase_bus, S_base, self._complex_np_dtype)

            Vf = Vbase_bus[Branch_f_bus]
            Vt = Vbase_bus[Branch_t_bus]

            Branch_y_series_from = (
                Branch_y_series_from * (Vf ** 2 / S_base)
            ).astype(self._complex_np_dtype, copy=False)

            Branch_y_series_to = (
                Branch_y_series_to * (Vt ** 2 / S_base)
            ).astype(self._complex_np_dtype, copy=False)

            Branch_y_series_ft = (
                Branch_y_series_ft * (Vf * Vt / S_base)
            ).astype(self._complex_np_dtype, copy=False)

            Branch_y_shunt_from = (
                Branch_y_shunt_from * (Vf ** 2 / S_base)
            ).astype(self._complex_np_dtype, copy=False)

            Branch_y_shunt_to = (
                Branch_y_shunt_to * (Vt ** 2 / S_base)
            ).astype(self._complex_np_dtype, copy=False)

            Y_Lines = (
                Y_Lines * (Vf ** 2 / S_base)
            ).astype(self._complex_np_dtype, copy=False)

            Y_C_Lines = (
                Y_C_Lines * (Vf ** 2 / S_base)
            ).astype(self._real_np_dtype, copy=False)

            u_start = u_si_to_pu_per_bus(u_start, Vbase_bus, self._complex_np_dtype)
            u_newton = u_si_to_pu_per_bus(u_newton, Vbase_bus, self._complex_np_dtype)

            S_start = s_si_to_pu(S_start, S_base, self._complex_np_dtype)
            S_newton = s_si_to_pu(S_newton, S_base, self._complex_np_dtype)

        elif not self._no_cache_dense_ybus and not reuse_shared_ybus:
            Ybus = Ybus_si.astype(self._complex_np_dtype, copy=False)

        V_start = np.stack(
            [np.abs(u_start).astype(self._real_np_dtype), np.angle(u_start).astype(self._real_np_dtype)],
            axis=1
        )
        V_newton = np.stack(
            [np.abs(u_newton).astype(self._real_np_dtype), np.angle(u_newton).astype(self._real_np_dtype)],
            axis=1
        )

        row: Dict[str, Any] = {
            "N": N,
            "nl": nl,
            "S_base": to_t(np.array(S_base, dtype=self._real_np_dtype), dtype=self._real_torch_dtype),
            "U_base": to_t(np.array(U_base_scalar, dtype=self._real_np_dtype), dtype=self._real_torch_dtype),

            "bus_type": to_t(bus_typ, dtype=torch.int64),
            "vn_kv": to_t(vn_kv, dtype=self._real_torch_dtype),
            "V_base_bus": to_t(Vbase_bus.astype(self._real_np_dtype), dtype=self._real_torch_dtype),
            "vn_log": to_t(np.log10(vn_kv.astype(np.float32) + 1e-9), dtype=torch.float32),
            "Y_shunt_bus": to_t(Y_shunt_bus, dtype=self._complex_torch_dtype),

            "Branch_f_bus": to_t(Branch_f_bus, dtype=torch.int64),
            "Branch_t_bus": to_t(Branch_t_bus, dtype=torch.int64),
            "Branch_status": to_t(Branch_status, dtype=torch.int8),
            "Branch_tau": to_t(Branch_tau, dtype=self._real_torch_dtype),
            "Branch_shift_deg": to_t(Branch_shift_deg, dtype=self._real_torch_dtype),

            "Branch_y_series_from": to_t(Branch_y_series_from, dtype=self._complex_torch_dtype),
            "Branch_y_series_to": to_t(Branch_y_series_to, dtype=self._complex_torch_dtype),
            "Branch_y_series_ft": to_t(Branch_y_series_ft, dtype=self._complex_torch_dtype),
            "Branch_y_shunt_from": to_t(Branch_y_shunt_from, dtype=self._complex_torch_dtype),
            "Branch_y_shunt_to": to_t(Branch_y_shunt_to, dtype=self._complex_torch_dtype),

            "Is_trafo": to_t(Is_trafo, dtype=torch.int8),
            "Branch_hv_is_f": to_t(Branch_hv_is_f, dtype=torch.int8),
            "Branch_n": to_t(Branch_n, dtype=self._real_torch_dtype),

            "Y_Lines": to_t(Y_Lines, dtype=self._complex_torch_dtype),
            "Y_C_Lines": to_t(Y_C_Lines, dtype=self._real_torch_dtype),

            "U_start": to_t(u_start, dtype=self._complex_torch_dtype),
            "U_newton": to_t(u_newton, dtype=self._complex_torch_dtype),
            "V_start": to_t(V_start, dtype=self._real_torch_dtype),
            "V_newton": to_t(V_newton, dtype=self._real_torch_dtype),
            "S_start": to_t(S_start, dtype=self._complex_torch_dtype),
            "S_newton": to_t(S_newton, dtype=self._complex_torch_dtype),
        }
        if not self._no_cache_dense_ybus:
            if reuse_shared_ybus:
                # Reuse the cached tensor. Tensors returned by __getitem__ are
                # only READ by the collate function (block_diag / stacking)
                # and the model forward, neither of which mutates Ybus -- so
                # sharing one tensor across all dataset rows is safe.
                row["Ybus"] = self._shared_ybus_tensor
            else:
                ybus_t = to_t(Ybus, dtype=self._complex_torch_dtype)
                row["Ybus"] = ybus_t
                if self._share_ybus and self._shared_ybus_tensor is None:
                    self._shared_ybus_tensor = ybus_t
                    print(
                        f"[Dataset]   share_ybus=True: cached shared Ybus from "
                        f"first row (N={N}, |Y|_pu max={ybus_t.abs().max():.3e}); "
                        f"subsequent rows will skip dense Y reconstruction"
                    )

        # share_grid first-row population: cache every grid-only tensor so
        # subsequent calls take the fast path at the top of this function.
        # Triggered only on the very first build (cache empty).
        if self._share_grid and not self._shared_grid_tensors:
            grid_field_names = [
                "bus_type", "vn_kv", "vn_log", "V_base_bus", "Y_shunt_bus",
                "Branch_f_bus", "Branch_t_bus", "Branch_status",
                "Branch_tau", "Branch_shift_deg",
                "Branch_y_series_from", "Branch_y_series_to", "Branch_y_series_ft",
                "Branch_y_shunt_from", "Branch_y_shunt_to",
                "Is_trafo", "Branch_hv_is_f", "Branch_n",
                "Y_Lines", "Y_C_Lines",
            ]
            if not self._no_cache_dense_ybus:
                grid_field_names.append("Ybus")
            for name in grid_field_names:
                if name in row:
                    self._shared_grid_tensors[name] = row[name]
            self._shared_grid_numpy["Vbase_bus"] = Vbase_bus
            self._shared_grid_numpy["bus_typ"] = bus_typ
            print(
                f"[Dataset]   share_grid=True: cached "
                f"{len(self._shared_grid_tensors)} grid-only tensors from "
                f"first row (Ybus + Branch_y_* + topology + vn_kv + ...); "
                f"subsequent rows skip parquet decode + per-unit math for "
                f"these fields"
            )

        row = self._attach_nr_trajectory(row, r, available_columns, Vbase_bus, N)
        return self._attach_extra_columns(row, r, available_columns)

    def _print_sbase_diagnostic(self, path, S_base_raw: float) -> None:
        """One-shot diagnostic: warn if the parquet's S_base disagrees with the
        target rebase value. Helps users notice scale mismatches that would
        otherwise silently break training (Armijo reject / pure-PINN diverge).
        """
        print(f"[Dataset] {path}")
        print(f"[Dataset]   parquet S_base = {S_base_raw:.3e} VA "
              f"(= {S_base_raw/1e6:.4g} MVA)")
        if self._per_unit and self._target_S_base is not None:
            if abs(S_base_raw - self._target_S_base) > 1.0:
                ratio = S_base_raw / self._target_S_base
                print(f"[Dataset]   per-unit S_base REBASED to "
                      f"{self._target_S_base:.3e} VA "
                      f"(= {self._target_S_base/1e6:.4g} MVA); "
                      f"S_pu/Y_pu scaled by {ratio:.3g}x")
            else:
                print(f"[Dataset]   per-unit S_base matches target "
                      f"({self._target_S_base/1e6:.4g} MVA); no rebase")
        elif self._per_unit:
            print(f"[Dataset]   per-unit conversion uses parquet S_base as-is "
                  f"(target_S_base=None)")
        else:
            print(f"[Dataset]   per_unit=False; S_base unused in conversion")

    def _load_eager_rows(self, paths: List[Union[str, Path]]) -> None:
        all_dfs = []

        for path in paths:
            print(f"Processing file {path} ...")
            present_columns = self._present_columns(pq.ParquetFile(str(path)).schema.names)
            df = pd.read_parquet(path, columns=present_columns, engine="pyarrow")

            binary_cols = [c for c in self._binary_cols_master if c in df.columns]
            df = decode_columns_mp_columnwise(df, binary_cols)
            print(f"Decoded columns from {path}: {binary_cols}")
            print(f"Raw shape: {df.shape}")

            if "S_base" in df.columns and len(df) > 0:
                self._print_sbase_diagnostic(path, float(df.iloc[0]["S_base"]))

            if "u_newton" in df.columns:
                keep_mask = ~df["u_newton"].map(_all_zero_complex_fast).to_numpy()
                if not keep_mask.all():
                    removed = int((~keep_mask).sum())
                    df = df.loc[keep_mask].reset_index(drop=True)
                    print(f"Removed {removed} diverged rows -> {df.shape}")

            all_dfs.append(df)

        if len(all_dfs) == 0:
            raise ValueError("No parquet data loaded.")

        if len(all_dfs) == 1:
            df = all_dfs[0]
        else:
            df = pd.concat(all_dfs, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Final combined dataset shape: {df.shape}")

        available_columns = set(df.columns)
        for _, r in df.iterrows():
            self.rows.append(self._build_row_dict(r, available_columns))

    def _build_lazy_row_group_index(self, paths: List[Union[str, Path]]) -> None:
        total_valid = 0
        total_removed = 0

        for file_idx, path in enumerate(paths):
            print(f"Indexing parquet lazily from {path} ...")
            parquet_file = pq.ParquetFile(str(path))
            self._parquet_files.append(parquet_file)

            # One-shot S_base diagnostic: read just the S_base column from
            # the first row group so we can warn about scale mismatches
            # without loading the heavy binary columns.
            if "S_base" in parquet_file.schema.names and parquet_file.num_row_groups > 0:
                sb_table = parquet_file.read_row_group(0, columns=["S_base"])
                sb_list = sb_table.column("S_base").to_pylist()
                if sb_list:
                    self._print_sbase_diagnostic(path, float(sb_list[0]))

            file_valid = 0
            file_removed = 0

            for row_group_idx in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(
                    row_group_idx,
                    columns=["bus_number", "branch_number", "u_newton"],
                )
                data = table.to_pydict()

                keep_positions = []
                signatures = []
                for row_pos, encoded_u_newton in enumerate(data["u_newton"]):
                    if _all_zero_complex_fast(npy_bytes_to_ndarray(encoded_u_newton)):
                        file_removed += 1
                        total_removed += 1
                        continue

                    keep_positions.append(row_pos)
                    signatures.append(
                        (
                            int(data["bus_number"][row_pos]),
                            int(data["branch_number"][row_pos]),
                        )
                    )

                if not keep_positions:
                    continue

                keep_positions_arr = np.asarray(keep_positions, dtype=np.int16)
                signatures_arr = np.asarray(signatures, dtype=np.int32)
                total_valid += len(keep_positions_arr)
                file_valid += len(keep_positions_arr)

                self._row_group_metas.append(
                    {
                        "file_idx": file_idx,
                        "row_group_idx": row_group_idx,
                        "keep_positions": keep_positions_arr,
                        "signatures": signatures_arr,
                    }
                )
                self._row_group_ends.append(total_valid)

            print(
                f"Indexed {file_valid} valid rows from {path}"
                f" across {parquet_file.num_row_groups} row groups"
                f" (removed {file_removed} diverged rows)"
            )

        if total_valid == 0:
            raise ValueError("No valid parquet rows remained after filtering diverged rows.")

        self._num_rows = total_valid
        print(
            f"Final combined dataset shape: ({self._num_rows}, streamed rows)"
            f" | removed {total_removed} diverged rows total"
        )

    def _decode_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self._binary_cols_master:
            if col in df.columns:
                df[col] = [npy_bytes_to_ndarray(b) for b in df[col].values]
        return df

    def _load_row_group_rows(self, meta_idx: int) -> List[Dict[str, Any]]:
        meta = self._row_group_metas[meta_idx]
        parquet_file = self._parquet_files[meta["file_idx"]]
        present_columns = self._present_columns(parquet_file.schema.names)

        table = parquet_file.read_row_group(
            meta["row_group_idx"],
            columns=present_columns,
        )
        df = table.to_pandas()
        df = self._decode_binary_columns(df)

        keep_positions = meta["keep_positions"]
        if len(keep_positions) != len(df):
            df = df.iloc[keep_positions].reset_index(drop=True)

        available_columns = set(df.columns)
        return [self._build_row_dict(r, available_columns) for _, r in df.iterrows()]

    def _get_lazy_meta_index(self, idx: int) -> int:
        return bisect_right(self._row_group_ends, idx)

    def _get_cached_row_group(self, meta_idx: int) -> List[Dict[str, Any]]:
        cached = self._row_group_cache.get(meta_idx)
        if cached is not None:
            self._row_group_cache.move_to_end(meta_idx)
            return cached

        rows = self._load_row_group_rows(meta_idx)
        self._row_group_cache[meta_idx] = rows
        self._row_group_cache.move_to_end(meta_idx)

        while len(self._row_group_cache) > self._row_group_cache_size:
            self._row_group_cache.popitem(last=False)

        return rows

    def get_signature(self, idx: int):
        if self.rows is not None:
            row = self.rows[idx]
            return int(row["N"]), int(row["nl"])

        meta_idx = self._get_lazy_meta_index(idx)
        prev_end = 0 if meta_idx == 0 else self._row_group_ends[meta_idx - 1]
        within = idx - prev_end
        signature = self._row_group_metas[meta_idx]["signatures"][within]
        return int(signature[0]), int(signature[1])

    def __len__(self) -> int:
        if self.rows is not None:
            return len(self.rows)
        return self._num_rows

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.rows is not None:
            return self.rows[idx]

        meta_idx = self._get_lazy_meta_index(idx)
        prev_end = 0 if meta_idx == 0 else self._row_group_ends[meta_idx - 1]
        within = idx - prev_end
        rows = self._get_cached_row_group(meta_idx)
        return rows[within]
