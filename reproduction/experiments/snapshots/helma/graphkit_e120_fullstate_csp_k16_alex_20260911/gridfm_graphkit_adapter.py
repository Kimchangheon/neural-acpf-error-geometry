"""Run the released `gridfm_graphkit` model instead of our local re-implementation.

LUMINA and GridSFM are evaluated as their authors published them, but "GridFM"
in this project has been `train_valid_test_gridfm.GridFMHeteroSurrogate`, a local
model written in GridFM's architectural style. That asymmetry makes the
comparison unfair to GridFM, and this module removes it by driving
`gridfm_graphkit.models.gnn_heterogeneous_gns.GNS_heterogeneous` directly.

What the released model does that the local mirror does not:

* a physics correction injected at **every** layer --- branch flows are computed
  from the running (Vm, Va), aggregated into node injections, and fed back
  through `physics_mlp`;
* `bound_with_sigmoid` on |V| between the per-bus band, and (OPF only) on Pg
  between the generator limits, instead of a fixed `clamp`;
* a generator head `mlp_gen -> Pg`;
* `mask_dict`-driven substitution: quantities that are *known* for a bus type
  are pinned to their input values rather than predicted.

One asymmetry this does **not** fix: `gridfm_graphkit` ships no released
weights (no checkpoint file, no HuggingFace reference anywhere in the repo), so
GridFM remains scratch-only while LUMINA and GridSFM have public checkpoints.

Column layouts are byte-identical between this project and
`gridfm_graphkit.datasets.globals`, so bus/gen features transfer unchanged. The
one real conversion is the edge attribute: we carry the Y-bus off-diagonal only
(2 columns), while the released model needs the per-branch admittance split
across 10 columns. `validate_edge_attr` re-assembles a Y-bus from that split and
compares it against the dataset's own, which is the check that this conversion
is right.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# Edge feature indices (gridfm_graphkit.datasets.globals)
P_E, Q_E = 0, 1
YFF_TT_R, YFF_TT_I = 2, 3
YFT_TF_R, YFT_TF_I = 4, 5
TAP, ANG_MIN, ANG_MAX, RATE_A = 6, 7, 8, 9
EDGE_DIM = 10

# Bus / gen indices are identical to train_valid_test_gridfm's, which is why
# those feature builders can be reused verbatim.
PD_H, QD_H, QG_H, VM_H, VA_H = 0, 1, 2, 3, 4
PQ_H, PV_H, REF_H = 5, 6, 7
MIN_VM_H, MAX_VM_H, MIN_QG_H, MAX_QG_H = 8, 9, 10, 11
GS, BS, VN_KV = 12, 13, 14
PG_H, MIN_PG, MAX_PG, C0_H, C1_H, C2_H = 0, 1, 2, 3, 4, 5
GEN_DIM = 6
BUS_DIM = 15


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #

def make_args(task_name: str = "PowerFlow", hidden_size: int = 48,
              num_layers: int = 12, attention_head: int = 8,
              dropout: float = 0.0) -> SimpleNamespace:
    """The nested config `GNS_heterogeneous.__init__` reads.

    Defaults match `gridfm-graphkit/examples/config/HGNS_*_case57.yaml`, which
    are also the settings the local mirror was run with, so capacity is matched.
    """
    if task_name not in ("PowerFlow", "OptimalPowerFlow", "StateEstimation"):
        raise ValueError(f"unknown task_name {task_name!r}")
    return SimpleNamespace(
        task=SimpleNamespace(task_name=task_name),
        model=SimpleNamespace(
            num_layers=num_layers, hidden_size=hidden_size,
            input_bus_dim=BUS_DIM, input_gen_dim=GEN_DIM,
            output_bus_dim=2, output_gen_dim=1,
            edge_dim=EDGE_DIM, attention_head=attention_head, dropout=dropout,
        ),
    )


def build_graphkit_model(task_name="PowerFlow", hidden_size=48, num_layers=12,
                         attention_head=8, dropout=0.0):
    from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous
    return GNS_heterogeneous(make_args(task_name, hidden_size, num_layers,
                                       attention_head, dropout))


def apply_graphkit_init_recipe(model, recipe: str = "default"):
    """Give HGNS the flat-start initialisation the working models already have.

    PIGNN zero-inits its heads so the first forward pass reproduces V_start;
    GridSFM fills head_V's bias with 1.0 so it starts at flat voltage.  HGNS
    contains no nn.init call at all -- every Linear, the terminal mlp_bus
    included, is left at PyTorch's default -- so its initial voltage is
    arbitrary.  PhysicsDecoderPF passes voltage straight through
    (Vm_out = bus_data_pred[:, VM_OUT]) and this adapter feeds raw per-unit
    values, so zeroing the last layer's weight and writing bias
    [1.0, 0.0] at (VM_OUT, VA_OUT) starts the model at 1.0 pu and 0 rad.
    Known entries are pinned by the model's own mask regardless.

    default    leave the released initialisation alone
    flathead   zero mlp_bus's last weight, bias -> flat voltage
    sd002      normal_(0, 0.02) weights, zero biases, everywhere
    both       sd002 followed by flathead
    """
    import torch
    import torch.nn as nn
    from gridfm_graphkit.datasets.globals import VM_OUT, VA_OUT

    if recipe == "default":
        return model

    if recipe in ("sd002", "both"):
        n = 0
        for m in model.modules():
            w = getattr(m, "weight", None)
            if isinstance(w, torch.Tensor) and w.dim() >= 2:
                nn.init.normal_(w, mean=0.0, std=0.02)
                b = getattr(m, "bias", None)
                if isinstance(b, torch.Tensor):
                    nn.init.zeros_(b)
                n += 1
        print(f"[init] graphkit normal_(0, 0.02) on {n} weight matrices", flush=True)

    if recipe in ("flathead", "both"):
        head = getattr(model, "mlp_bus", None)
        if head is None:
            raise SystemExit("apply_graphkit_init_recipe: model has no mlp_bus")
        cands = [m for m in head.modules()
                 if isinstance(getattr(m, "weight", None), torch.Tensor)
                 and m.weight.dim() == 2]
        if not cands:
            raise SystemExit("apply_graphkit_init_recipe: no weight matrix in mlp_bus")
        tgt = cands[-1]
        nn.init.zeros_(tgt.weight)
        if not isinstance(getattr(tgt, "bias", None), torch.Tensor):
            raise SystemExit("apply_graphkit_init_recipe: mlp_bus head has no bias, "
                             "so a flat start cannot be written")
        nn.init.zeros_(tgt.bias)
        with torch.no_grad():
            tgt.bias[VM_OUT] = 1.0
            tgt.bias[VA_OUT] = 0.0
        print(f"[init] graphkit mlp_bus head -> flat start "
              f"(out_dim {tuple(tgt.weight.shape)}, bias[VM]=1.0, bias[VA]=0.0)",
              flush=True)
    return model


def configure_neural_edge_transform(model, mode: str = "raw"):
    """Transform edge admittances only on their way into the neural encoder.

    The released GraphKit backbone deliberately reuses the same raw
    ``edge_attr`` for two different purposes: ``input_proj_edge`` embeds it for
    message passing, while ``ComputeBranchFlow`` uses it as physical
    admittance.  Mutating the HeteroData edge tensor would therefore make the
    branch-flow equations operate on signed-log admittance.  A forward pre-hook
    keeps the public model, its parameters, and its physical path untouched and
    transforms only columns 2:6 seen by the neural projection.
    """
    if mode not in ("raw", "signed_log"):
        raise ValueError(
            f"unknown GraphKit neural edge transform {mode!r}; expected "
            "'raw' or 'signed_log'"
        )
    old_handle = getattr(model, "_ppc_edge_transform_handle", None)
    if old_handle is not None:
        old_handle.remove()
        model._ppc_edge_transform_handle = None
    model._ppc_neural_edge_transform = mode
    if mode == "raw":
        return model

    def _signed_log_admittance(_module, inputs):
        edge_attr = inputs[0]
        transformed = edge_attr.clone()
        values = transformed[:, YFF_TT_R : YFT_TF_I + 1]
        transformed[:, YFF_TT_R : YFT_TF_I + 1] = (
            torch.sign(values) * torch.log1p(values.abs())
        )
        return (transformed, *inputs[1:])

    model._ppc_edge_transform_handle = model.input_proj_edge.register_forward_pre_hook(
        _signed_log_admittance
    )
    return model


# --------------------------------------------------------------------------- #
# per-branch admittance split
# --------------------------------------------------------------------------- #

def _branch_admittance(r, x, b_fr, b_to, tap, shift):
    """MATPOWER branch model, matching `opfdata_pipeline.build_ybus` exactly.

    Returns (Yff, Yft, Ytt, Ytf) as complex numpy arrays. The released model
    consumes one directed edge per end: the i->j edge carries (Yff, Yft) and the
    j->i edge carries (Ytt, Ytf), which is what its YFF_TT / YFT_TF naming means.
    """
    z = r + 1j * x
    ys = np.where(np.abs(z) > 0, 1.0 / np.where(np.abs(z) > 0, z, 1.0), 0.0 + 0j)
    tp = tap * np.exp(1j * shift)
    Yff = (ys + 1j * b_fr) / (np.abs(tp) ** 2)
    Ytt = ys + 1j * b_to
    Yft = -ys / np.conj(tp)
    Ytf = -ys / tp
    return Yff, Yft, Ytt, Ytf


def graphkit_branch_edges(hetero, device, dtype=torch.float32
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Directed bus-bus edges with the released model's 10-column attribute."""
    from opfdata_pipeline import AC_LINE, TRAFO

    src, dst, attrs = [], [], []
    for key, cols, is_trafo in ((("bus", "ac_line", "bus"), AC_LINE, False),
                                (("bus", "transformer", "bus"), TRAFO, True)):
        if key not in hetero.edge_types:
            continue
        store = hetero[key]
        ei = store.edge_index
        ea = getattr(store, "edge_attr", None)
        if ea is None or ei.numel() == 0:
            continue
        a = ea.detach().cpu().to(torch.float64).numpy()
        f = ei[0].detach().cpu().numpy()
        t = ei[1].detach().cpu().numpy()
        r, x = a[:, cols["r"]], a[:, cols["x"]]
        b_fr, b_to = a[:, cols["b_fr"]], a[:, cols["b_to"]]
        if is_trafo:
            tap = np.where(a[:, cols["tap"]] != 0, a[:, cols["tap"]], 1.0)
            shift = a[:, cols["shift"]]
        else:
            tap = np.ones_like(r)
            shift = np.zeros_like(r)
        rate = a[:, cols["rate_a"]] if "rate_a" in cols else np.zeros_like(r)
        amin = a[:, cols["angmin"]] if "angmin" in cols else np.zeros_like(r)
        amax = a[:, cols["angmax"]] if "angmax" in cols else np.zeros_like(r)

        Yff, Yft, Ytt, Ytf = _branch_admittance(r, x, b_fr, b_to, tap, shift)

        for (s_i, d_i, Yself, Ymut) in ((f, t, Yff, Yft), (t, f, Ytt, Ytf)):
            e = np.zeros((len(r), EDGE_DIM), dtype=np.float64)
            e[:, YFF_TT_R] = Yself.real
            e[:, YFF_TT_I] = Yself.imag
            e[:, YFT_TF_R] = Ymut.real
            e[:, YFT_TF_I] = Ymut.imag
            e[:, TAP] = tap
            e[:, ANG_MIN] = amin
            e[:, ANG_MAX] = amax
            e[:, RATE_A] = rate
            src.append(s_i); dst.append(d_i); attrs.append(e)

    if not attrs:
        return (torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0, EDGE_DIM), dtype=dtype, device=device))
    ei = torch.tensor(np.stack([np.concatenate(src), np.concatenate(dst)]),
                      dtype=torch.long, device=device)
    ea = torch.tensor(np.concatenate(attrs, axis=0), dtype=dtype, device=device)
    return ei, ea


def validate_edge_attr(hetero, Ybus_ref: torch.Tensor, rtol=1e-12) -> Dict[str, float]:
    """Re-assemble a Y-bus from the per-branch split and compare to the source.

    The split is the only non-trivial part of this adapter, and a wrong split
    would degrade the model silently rather than raise. Shunts are added here
    because they live on buses, not branches.
    """
    # float64 here: a float32 round-trip alone costs ~1e-8 relative, which would
    # mask nothing but would also make a real formula error hard to see.
    ei, ea = graphkit_branch_edges(hetero, torch.device("cpu"), dtype=torch.float64)
    n = int(hetero["bus"].x.shape[0])
    Y = torch.zeros((n, n), dtype=torch.complex128)
    a = ea.to(torch.float64)
    yself = torch.complex(a[:, YFF_TT_R], a[:, YFF_TT_I])
    ymut = torch.complex(a[:, YFT_TF_R], a[:, YFT_TF_I])
    Y.index_put_((ei[0], ei[0]), yself, accumulate=True)
    Y.index_put_((ei[0], ei[1]), ymut, accumulate=True)

    if "shunt" in hetero.node_types and hetero["shunt"].x.numel():
        sx = hetero["shunt"].x.detach().cpu().to(torch.float64)
        sb = hetero["shunt", "shunt_link", "bus"].edge_index[1].detach().cpu()
        for k in range(sx.shape[0]):
            b = int(sb[k])
            Y[b, b] += torch.complex(sx[k, 1], sx[k, 0])

    ref = Ybus_ref.detach().cpu().to(torch.complex128)
    err = (Y - ref).abs()
    rel = float(err.max().item() / max(ref.abs().max().item(), 1e-12))
    return {"max_abs_err": float(err.max().item()), "rel_err": rel,
            "ok": bool(rel <= rtol)}


# --------------------------------------------------------------------------- #
# batch construction
# --------------------------------------------------------------------------- #

def _bus_mask(x_bus: torch.Tensor, task_name: str) -> torch.Tensor:
    """Which bus quantities the model must predict rather than copy.

    Power flow: |V| is known at PV and ref buses, the angle only at ref. OPF:
    the whole voltage profile is free, since the optimiser chooses it.
    """
    n = x_bus.shape[0]
    m = torch.zeros((n, BUS_DIM), dtype=torch.bool, device=x_bus.device)
    if task_name == "OptimalPowerFlow":
        m[:, VM_H] = True
        m[:, VA_H] = True
        return m
    is_pq = x_bus[:, PQ_H] > 0.5
    is_ref = x_bus[:, REF_H] > 0.5
    m[:, VM_H] = is_pq
    m[:, VA_H] = ~is_ref
    return m


def to_graphkit_batch(batch: Dict[str, torch.Tensor], device,
                      task_name: str = "PowerFlow",
                      feature_transform: str = "signed_log"):
    """OPFData batch -> the HeteroData the released model's `forward` expects."""
    from torch_geometric.data import HeteroData
    from opfdata_pipeline import to_gridfm_inputs

    # Bus/gen features are already in gridfm_graphkit's exact column layout.
    x_dict, ei_dict, _ = to_gridfm_inputs(batch, device, feature_transform)
    x_bus = x_dict["bus"]
    x_gen = x_dict["gen"][:, :GEN_DIM]  # released model reads 6, not 7 (no G_ON)

    hetero = batch["hetero"].to(device)
    bus_ei, bus_ea = graphkit_branch_edges(hetero, device)

    d = HeteroData()
    d["bus"].x = x_bus
    d["gen"].x = x_gen
    d["bus", "connects", "bus"].edge_index = bus_ei
    d["bus", "connects", "bus"].edge_attr = bus_ea
    for rel in (("gen", "connected_to", "bus"), ("bus", "connected_to", "gen")):
        d[rel].edge_index = ei_dict[rel]
        d[rel].edge_attr = None

    # The physics decoders additionally read bus-type masks by name ("PV",
    # "REF"): they recover Qg from the nodal reactive balance at PV and slack
    # buses, where it is not a free variable.
    d.mask_dict = {
        "bus": _bus_mask(x_bus, task_name),
        "gen": torch.ones((x_gen.shape[0], GEN_DIM), dtype=torch.bool, device=device),
        "PQ": x_bus[:, PQ_H] > 0.5,
        "PV": x_bus[:, PV_H] > 0.5,
        "REF": x_bus[:, REF_H] > 0.5,
    }
    return d


def forward_graphkit(model, batch, device, task_name="PowerFlow",
                     feature_transform="signed_log", want_pred6=False):
    """Returns (V [1,N,2] as (mag, angle), gen_pred, gen_bus[, pred6]).

    `pred6` is the [VM, VA, PG, QG, PD, QD] layout that GridFM's own
    `MaskedReconstructionMSE` consumes. Here PG and QG come from the model's
    physics decoder rather than from an extra regression head, so this is a
    more faithful input to that loss than the local mirror could provide.
    """
    d = to_graphkit_batch(batch, device, task_name, feature_transform)
    out = model(d)
    # For PF/OPF the bus output comes from the physics decoder as
    # [Vm, Va, Pg, Qg]; only the first two are the voltage state.
    bus_out = out["bus"] if isinstance(out, dict) else out
    mag = bus_out[:, 0]
    ang = torch.atan2(torch.sin(bus_out[:, 1]), torch.cos(bus_out[:, 1]))
    V = torch.stack([mag, ang], dim=-1).unsqueeze(0)
    gen_pred = out.get("gen") if isinstance(out, dict) else None
    gen_bus = d["gen", "connected_to", "bus"].edge_index[1]
    if not want_pred6:
        return V, gen_pred, gen_bus
    x_bus = d["bus"].x
    pd_qd = x_bus[:, [PD_H, QD_H]]          # known inputs, masked out of the loss
    pg_qg = bus_out[:, 2:4] if bus_out.shape[1] >= 4 else torch.zeros_like(pd_qd)
    pred6 = torch.cat([torch.stack([mag, ang], dim=-1), pg_qg, pd_qd], dim=-1)
    return V, gen_pred, gen_bus, pred6


# --------------------------------------------------------------------------- #
# parquet (PPC branch-row) pipeline
# --------------------------------------------------------------------------- #

def graphkit_branch_edges_parquet(batch: Dict[str, torch.Tensor], device,
                                  dtype=torch.float32
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same 10-column edge attribute, built from the PPC branch-row parquet.

    Important: with `--PER_UNIT` the loader has *already* converted the branch
    admittances to pu with the local bus bases
    (`Branch_y_series_from *= Vf**2/S_base`, `..._ft *= Vf*Vt/S_base`, and so on
    in `Dataset_optimized_complex_columns`). Applying the bus bases again here
    is wrong by a factor of Vbase^2/S_base -- an early version of this function
    did exactly that and missed the Y-bus by a relative 7.5e+2, which is what
    `validate_edge_attr_parquet` is for.

    In pu the Vbase re-referencing the SI path performs collapses to an
    identity: y_ft_pu == y_from_pu and y_tf_pu == y_to_pu, so the from/to
    series values are used directly.
    """
    def flat(k):
        v = batch[k]
        return v.squeeze(0) if v.dim() > 1 else v

    f = flat("Branch_f_bus").to(torch.long).cpu().numpy()
    t = flat("Branch_t_bus").to(torch.long).cpu().numpy()
    status = flat("Branch_status").to(torch.float64).cpu().numpy()
    tau = flat("Branch_tau").to(torch.float64).cpu().numpy()
    shift = np.deg2rad(flat("Branch_shift_deg").to(torch.float64).cpu().numpy())
    y_from = flat("Branch_y_series_from").to(torch.complex128).cpu().numpy()
    y_to = flat("Branch_y_series_to").to(torch.complex128).cpu().numpy()
    ysh_f = flat("Branch_y_shunt_from").to(torch.complex128).cpu().numpy()
    ysh_t = flat("Branch_y_shunt_to").to(torch.complex128).cpu().numpy()
    rate = (flat("Branch_rate_a").to(torch.float64).cpu().numpy()
            if "Branch_rate_a" in batch else np.zeros_like(tau))

    a = tau * np.exp(1j * shift)
    Yff = (y_from + ysh_f / 2.0) / (a * np.conj(a))
    Ytt = (y_to + ysh_t / 2.0)
    Yft = -y_from / np.conj(a)
    Ytf = -y_to / a

    live = status != 0
    src, dst, attrs = [], [], []
    for (s_i, d_i, Yself, Ymut) in ((f, t, Yff, Yft), (t, f, Ytt, Ytf)):
        e = np.zeros((len(f), EDGE_DIM), dtype=np.float64)
        e[:, YFF_TT_R] = Yself.real
        e[:, YFF_TT_I] = Yself.imag
        e[:, YFT_TF_R] = Ymut.real
        e[:, YFT_TF_I] = Ymut.imag
        e[:, TAP] = tau
        e[:, RATE_A] = rate
        src.append(s_i[live]); dst.append(d_i[live]); attrs.append(e[live])

    ei = torch.tensor(np.stack([np.concatenate(src), np.concatenate(dst)]),
                      dtype=torch.long, device=device)
    ea = torch.tensor(np.concatenate(attrs, axis=0), dtype=dtype, device=device)
    return ei, ea


def validate_edge_attr_parquet(batch: Dict[str, torch.Tensor],
                               rtol=1e-9) -> Dict[str, float]:
    """Re-assemble the pu Y-bus from the parquet split and compare to the batch's."""
    ei, ea = graphkit_branch_edges_parquet(batch, torch.device("cpu"),
                                           dtype=torch.float64)
    ref = batch["Ybus"]
    # The collate now hands out a sparse block-diagonal Y-bus. This validator
    # compares full matrices, so densify here -- it runs once per grid on a
    # handful of samples, where the dense form is cheap, and squeeze/max are
    # unimplemented for the sparse backends anyway.
    if ref.is_sparse:
        ref = ref.to_dense()
    ref = (ref.squeeze(0) if ref.dim() > 2 else ref).cpu().to(torch.complex128)
    n = ref.shape[0]
    Y = torch.zeros((n, n), dtype=torch.complex128)
    a = ea.to(torch.float64)
    Y.index_put_((ei[0], ei[0]), torch.complex(a[:, YFF_TT_R], a[:, YFF_TT_I]),
                 accumulate=True)
    Y.index_put_((ei[0], ei[1]), torch.complex(a[:, YFT_TF_R], a[:, YFT_TF_I]),
                 accumulate=True)

    ysh = batch.get("Y_shunt_bus")
    if ysh is not None:
        # already pu under --PER_UNIT, like the branch values
        ysh = (ysh.squeeze(0) if ysh.dim() > 1 else ysh).cpu().to(torch.complex128)
        Y += torch.diag(ysh)

    err = (Y - ref).abs()
    rel = float(err.max().item() / max(ref.abs().max().item(), 1e-12))
    return {"max_abs_err": float(err.max().item()), "rel_err": rel,
            "ok": bool(rel <= rtol)}


def to_graphkit_batch_parquet(batch: Dict[str, torch.Tensor], device,
                              task_name: str = "PowerFlow",
                              feature_transform: str = "signed_log",
                              vn_feature_mode: str = "log",
                              opf_space=None,
                              zero_unspecified_q: bool = False):
    """PPC branch-row parquet batch -> the HeteroData the released model expects.

    Mirrors `to_graphkit_batch` (OPFData) but takes its bus/gen features from the
    parquet builder and its edge attribute from
    `graphkit_branch_edges_parquet`, which carries the per-branch self term the
    released model needs to compute branch flows.
    """
    from torch_geometric.data import HeteroData
    # Imported here rather than at module scope: the parquet driver imports this
    # module, so a top-level import would be circular.
    from train_valid_test_gridfm import make_gridfm_inputs

    x_dict, ei_dict, _, _, _, _ = make_gridfm_inputs(
        batch,
        device,
        vn_feature_mode,
        feature_transform,
        opf_space=opf_space,
        zero_unspecified_q=zero_unspecified_q,
    )
    x_bus = x_dict["bus"]
    x_gen = x_dict["gen"][:, :GEN_DIM]      # released model reads 6 cols, no G_ON

    bus_ei, bus_ea = graphkit_branch_edges_parquet(batch, device)

    d = HeteroData()
    d["bus"].x = x_bus
    d["gen"].x = x_gen
    d["bus", "connects", "bus"].edge_index = bus_ei
    d["bus", "connects", "bus"].edge_attr = bus_ea
    for rel in (("gen", "connected_to", "bus"), ("bus", "connected_to", "gen")):
        d[rel].edge_index = ei_dict[rel]
        d[rel].edge_attr = None

    # Use the adapter's own bus mask, not the mirror's: for power flow the
    # unknowns follow bus type (|V| free only at PQ, angle free everywhere but
    # the slack), whereas the mirror predicts both at every bus. "PV"/"REF" are
    # read by name by the physics decoder to recover Qg where it is not free.
    d.mask_dict = {
        "bus": _bus_mask(x_bus, task_name),
        "gen": torch.ones((x_gen.shape[0], GEN_DIM), dtype=torch.bool, device=device),
        "PQ": x_bus[:, PQ_H] > 0.5,
        "PV": x_bus[:, PV_H] > 0.5,
        "REF": x_bus[:, REF_H] > 0.5,
    }
    # Exact, untransformed PF tensors for fused known-operator models.  These
    # are metadata, not learned features: signed-log bus inputs are useful to
    # the GNN but must never be used to define the physical residual.
    d.kol_Ybus = batch["Ybus"]
    d.kol_Sset = batch["S_start"]
    d.kol_bus_type = batch["bus_type"]
    d.kol_sizes = batch["sizes"]
    return d


def forward_graphkit_parquet(model, batch, device, task_name="PowerFlow",
                             feature_transform="signed_log",
                             vn_feature_mode="log", opf_space=None,
                             zero_unspecified_q=False, mask_known_v=False, anchor_args=None):
    """Returns V as [1, N, 2] = (magnitude, angle), matching the mirror's output."""
    d = to_graphkit_batch_parquet(batch, device, task_name, feature_transform,
                                  vn_feature_mode, opf_space,
                                  zero_unspecified_q=zero_unspecified_q)
    out = model(d)
    bus_out = out["bus"] if isinstance(out, dict) else out
    mag = bus_out[:, 0]
    ang = torch.atan2(torch.sin(bus_out[:, 1]), torch.cos(bus_out[:, 1]))
    V = torch.stack([mag, ang], dim=-1).unsqueeze(0)
    # Anchors first, masking second: masking enforces the specified quantities
    # and must have the last word over any reparametrisation.
    if anchor_args is not None:
        from pf_anchors import apply_anchors
        V = apply_anchors(V, batch, anchor_args)
    if mask_known_v:
        from pf_known_mask import apply_known_v
        V = apply_known_v(V, batch["bus_type"].reshape(-1), batch["V_start"])
    return V
