import os
import sys
import time
import math
import argparse
import atexit
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from GNSMsg_SelfAttention_armijo import GNSMsg_EdgeSelfAttn
from GNSMsg_SelfAttention_armijo_khop import GNSMsg_EdgeSelfAttnKHop
from GNSMsg_armijo import GNSMsg   # adapt separately if you still want the non-attention baseline

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag, ybus_matvec

from helper import MultiBucketBatchSampler, make_size_bucketing_loader
from known_operator_pf import (
    add_known_operator_args,
    build_known_operator,
    known_operator_tag,
    power_flow_residual_loss,
)
from helm_known_operator_pf import PIGNNHELMKOL, add_helm_kol_args
from grad_accum import GradAccumulator, add_grad_accum_args, resolve_accum_steps
from preload_dataset import add_preload_args, preload_split
from ddp_utils import (add_ddp_args, barrier, cleanup, make_loader,
                       quiet_non_main, reduce_sums, setup_ddp, unwrap, wrap_model)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs("./results/plots", exist_ok=True)
# Note: checkpoint dir is created later, after --ckpt_dir is parsed.


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Training script for branch-row pandapower parquet format")

parser.add_argument(
    "--ckpt_dir",
    type=str,
    default="./results/ckpt",
    help=(
        "Directory to write best-model checkpoints into. Default keeps the "
        "historical ./results/ckpt path (relative to working dir). For Alex "
        "runs prefer an absolute path under /home/vault (TB-scale quota), "
        "e.g. --ckpt_dir /home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ckpt/<run_group>"
    ),
)
parser.add_argument("--PINN", action="store_true", help="Enable PINN")
parser.add_argument("--BLOCK_DIAG", action="store_true", help="Use block diagonal batching")
parser.add_argument("--NORMALIZE", action="store_true")
parser.add_argument("--PER_UNIT", action="store_true")
parser.add_argument(
    "--target_S_base",
    type=float,
    default=None,
    help=(
        "Optional S_base (VA) for per-unit rebasing. If unset, keep the "
        "parquet's S_base as-is. Pass e.g. --target_S_base 1e8 to force "
        "100 MVA (normalizes LVN-style 1 MVA distribution data into the "
        "IEEE/PEGASE scale). Only effective when --PER_UNIT is set."
    ),
)
parser.add_argument(
    "--dataset_complex_dtype",
    type=str,
    default="complex64",
    choices=("complex64", "complex128"),
    help=(
        "Complex dtype used by ChanghunDataset when decoding raw parquet "
        "admittance/voltage/power fields. Use complex128 for LVN validation/"
        "test residual diagnostics; complex64 is the memory-efficient default "
        "for training."
    ),
)
parser.add_argument(
    "--share_ybus",
    action="store_true",
    help=(
        "Build the dense Ybus tensor from row 0 once and reuse it across "
        "every row in the dataset. Safe when all rows describe the SAME "
        "grid sampled under different load/generation perturbations -- which "
        "is the case for LVN and the IEEE perturbation parquets. Skips "
        "per-row Y reconstruction (~722^2 complex matmul) and removes the "
        "single biggest source of lazy-parquet disk thrashing."
    ),
)
parser.add_argument(
    "--share_grid",
    action="store_true",
    help=(
        "Stronger than --share_ybus: cache ALL grid-only tensors "
        "(Branch_y_*, vn_kv, bus_type, Y_shunt_bus, Y_Lines, Y_C_Lines, "
        "topology, transformer flags) from row 0 and reuse for every row. "
        "Implies --share_ybus. Skips parquet decode + tensor allocation + "
        "per-unit math for ~20 grid-only fields. Use when every row is the "
        "same grid under different load/generation perturbations."
    ),
)
parser.add_argument(
    "--vn_feat",
    action="store_true",
    help=(
        "Add per-bus vn_log (log10 of rated kV) as an extra bus feature. "
        "Enables bus_feat_extra_dim=1 so the model can distinguish voltage "
        "classes (3/20/110/380 kV) in multi-voltage grids like LVN Heo1. "
        "Without this flag all buses look identical after per-unit "
        "normalization (V=1.0 plateau pathology)."
    ),
)
parser.add_argument(
    "--bus_type_features",
    action="store_true",
    help=(
        "Append slack/PV/PQ one-hot features and a zero-injection flag to the "
        "neural bus input. Output masks and exact physics are unchanged."
    ),
)
parser.add_argument(
    "--heterogeneous_injection_features",
    action="store_true",
    help=(
        "Fuse separate GridSFM-style proxy-generator and signed PQ-load "
        "embeddings into every PIGNN stage. The bus/branch attention backbone "
        "and K stage-specific update heads are unchanged."
    ),
)
parser.add_argument(
    "--two_hop_attention_mode",
    type=str,
    default="none",
    choices=("none", "pre", "post"),
    help=(
        "Sparse distance-exactly-two attention shared over K. pre inserts "
        "before the one-hop stack; post inserts after it and before any "
        "post-global or RANGE context."
    ),
)
parser.add_argument(
    "--global_context_mode",
    type=str,
    default="none",
    choices=("none", "meanmax_pre", "attn_pre", "attn_post", "range_post"),
    help=(
        "Optional per-graph global context sidecar. meanmax_pre uses mean/max "
        "pooling before local edge attention; attn_pre uses one-query "
        "cross-attention before local attention; attn_post injects the same "
        "attentive token after local attention. range_post uses persistent "
        "RANGE-style master nodes with node-to-master-to-node attention after "
        "local processing. All modules are shared over K and avoid dense N x N "
        "attention."
    ),
)
parser.add_argument(
    "--global_context_gate_mode",
    type=str,
    default="scalar_tanh",
    choices=("scalar_tanh", "sdpa_sigmoid"),
    help=(
        "Gating inside attentive graph context. scalar_tanh is the original "
        "zero-initialized scalar residual branch. sdpa_sigmoid additionally "
        "uses dynamic head/channel-specific sigmoid gates after the attention "
        "weighted value and before W_O, while retaining the outer zero-init "
        "scalar for baseline-equivalent initialization."
    ),
)
parser.add_argument("--range_num_masters", type=int, default=1)
parser.add_argument("--range_master_dim", type=int, default=24)
parser.add_argument("--range_num_heads", type=int, default=8)
parser.add_argument(
    "--range_positional_encoding",
    type=str,
    default="hop_slack_rbf",
    choices=("hop_slack_rbf",),
)
parser.add_argument("--range_pe_dim", type=int, default=10)
parser.add_argument("--float64", action="store_true")
parser.add_argument('--mode', type=str, default="train_test", help='train_valid_test | train | valid | test')
parser.add_argument("--mag_ang_mse", action="store_true", help="Use |V| + wrapped-angle reporting")
parser.add_argument(
    "--mse_weight",
    type=float,
    default=0.0,
    help=(
        "Supervised weight in L = physics_weight * L_phys + "
        "mse_weight * L_MSE. "
        "Default 0.0 = pure PINN (original behaviour). "
        "Set e.g. --mse_weight 10.0 to replicate the student's combined loss "
        "that stabilizes training on stiff/multi-voltage grids like LVN Heo1 "
        "where pure PINN converges to a non-V_newton solution. "
        "Only effective when --PINN is set."
    ),
)
parser.add_argument(
    "--physics_weight",
    type=float,
    default=1.0,
    help=(
        "Outer weight on the model's accumulated physics loss. The PF corpus "
        "reference configuration uses MSE + 1e-2 * physics, i.e. "
        "--mse_weight 1 --physics_weight 0.01."
    ),
)
parser.add_argument(
    "--correction_target_norm",
    type=str,
    default="none",
    choices=("none", "bus", "voltage_level"),
    help=(
        "Whiten the supervised correction target (V_newton - V_start) in the "
        "loss. 'bus' uses one train-set std per bus and channel; "
        "'voltage_level' pools those statistics over buses with the same vn_log. "
        "Predictions and reported RMSE remain in physical pu/radian units."
    ),
)
parser.add_argument(
    "--correction_target_norm_eps",
    type=float,
    default=1e-5,
    help="Minimum train-set std used by --correction_target_norm.",
)

parser.add_argument(
    '--model',
    type=str,
    default="GNSMsg_EdgeSelfAttn",
    help='GNSMsg_EdgeSelfAttn | GNSMsg_EdgeSelfAttnKHop | PIGNN_HELM_KOL',
)
parser.add_argument("--d", type=int, default=4)
parser.add_argument("--d_hi", type=int, default=16)
parser.add_argument("--num_attn_layers", type=int, default=1)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--khop_K", type=int, default=3)
parser.add_argument("--khop_sigma", type=float, default=1.5)
parser.add_argument("--khop_norm", type=str, default="row", choices=("row", "sym", "col"))
parser.add_argument("--khop_source", type=str, default="yabs", choices=("yabs", "adj"))

parser.add_argument("--K", type=int, default=40)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument("--use_armijo", action="store_true")
parser.add_argument(
    "--armijo_mode",
    type=str,
    default="fixed",
    choices=("fixed", "geometric", "geometric_safe", "reject"),
)
parser.add_argument("--armijo_rho", type=float, default=0.5)
parser.add_argument("--armijo_c1", type=float, default=1e-4)
parser.add_argument("--armijo_max_backtracks", type=int, default=5)
parser.add_argument("--armijo_min_alpha", type=float, default=0.0625)
parser.add_argument(
    "--armijo_norm", type=str, default="inf", choices=("inf", "rms"),
    help=(
        "Norm the Armijo acceptance test reduces the AC mismatch with. "
        "'inf' is the worst bus in the batch, so one pathological bus can "
        "veto the step for every scenario; 'rms' averages instead."
    ),
)
parser.add_argument("--vlimit", action="store_true")
parser.add_argument('--DthetaMax', type=float, default=0.3)
parser.add_argument('--DvmFrac', type=float, default=0.1)
parser.add_argument(
    "--solver_update_mode",
    choices=("direct", "physics_preconditioner", "physics_projected_hybrid"),
    default="direct",
    help=(
        "direct uses the historical learned voltage corrections; "
        "physics_preconditioner makes PIGNN predict positive diagonal scales "
        "for an author-compatible DPF direction; physics_projected_hybrid "
        "combines a full PIGNN proposal with DPF and projects it to a physical "
        "descent half-space."
    ),
)
parser.add_argument(
    "--preconditioner_vm_step", type=float, default=0.003377,
    help="Base multiplier for the PQ-magnitude DPF direction.",
)
parser.add_argument(
    "--preconditioner_va_step", type=float, default=0.003377,
    help="Base multiplier for the PV/PQ-angle DPF direction.",
)
parser.add_argument("--preconditioner_log_clip", type=float, default=5.0)
parser.add_argument(
    "--preconditioner_optimizer",
    choices=("author_adam", "gd"),
    default="author_adam",
    help="DPF base direction; author_adam matches the reference repository.",
)
parser.add_argument("--preconditioner_beta1", type=float, default=0.979681)
parser.add_argument("--preconditioner_beta2", type=float, default=0.963442)
parser.add_argument("--preconditioner_eps", type=float, default=1e-8)
parser.add_argument("--physics_loss_form", type=str, default="mse", choices=("mse", "huber", "logcosh"))
parser.add_argument("--physics_residual_norm", type=str, default="none", choices=("none", "setpoint", "graph"))
parser.add_argument(
    "--residual_feature_norm", type=str, default="none",
    choices=("none", "signed_log", "ybus", "dual"),
    help=(
        "Rescale DP/DQ before they enter the input projection. "
        "--physics_residual_norm only rescales the loss, so on an "
        "ill-conditioned grid the raw mismatch still reaches in_proj."
    ),
)
parser.add_argument(
    "--edge_feature_norm", type=str, default="none",
    choices=("none", "signed_log", "diagonal", "dual"),
    help=(
        "Transform only the six admittance-derived attention edge channels. "
        "The exact branch tensors and Y-bus used by the AC equations remain raw."
    ),
)
parser.add_argument(
    "--relative_stiffness_feature", action="store_true",
    help=(
        "Append log(|Y_ii|) centered by the geometric mean within each graph. "
        "This affects only the neural bus input; exact AC equations remain raw."
    ),
)
parser.add_argument(
    "--exact_physics_weight", type=float, default=0.0,
    help=(
        "Weight of an additional final-state AC-PF loss evaluated from the "
        "dataset's exact Y-bus. Use with --dataset_complex_dtype complex128 "
        "for the LVN conditioning diagnostic."
    ),
)
parser.add_argument(
    "--exact_physics_residual_norm", type=str, default="local_ybus",
    choices=("none", "local_ybus"),
    help="Residual scaling for --exact_physics_weight.",
)
add_grad_accum_args(parser)
add_preload_args(parser)
add_ddp_args(parser)
parser.add_argument("--physics_norm_eps", type=float, default=1e-6)
parser.add_argument("--physics_huber_delta", type=float, default=1.0)
parser.add_argument("--physics_final_weight", type=float, default=0.0)
parser.add_argument('--train_ratio', type=float, default=0.3333)
parser.add_argument('--valid_ratio', type=float, default=0.3333)
parser.add_argument("--max_train_samples", type=int, default=0, help="Cap train split size after random split; 0 disables")
parser.add_argument("--max_valid_samples", type=int, default=0, help="Cap valid split size after random split; 0 disables")
parser.add_argument("--max_test_samples", type=int, default=0, help="Cap test split size after random split; 0 disables")

parser.add_argument('--weight_init', type=str, default="sd0.02")
parser.add_argument('--bias_init', type=float, default=0.0)
parser.add_argument(
    "--preserve_zero_heads",
    action="store_true",
    help="Keep model output heads at their constructor zero-init instead of overwriting them in global init.",
)
parser.add_argument('--weight_decay', type=float, default=1e-3)

parser.add_argument('--lr_scheduler', type=str, default="default", help='default | CosineAnnealingLR')
parser.add_argument('--cosineRestartEpoch', type=int, default=20)

parser.add_argument("--BATCH", type=int, default=16)
parser.add_argument("--EPOCHS", type=int, default=20)
parser.add_argument("--LR", type=float, default=1e-4)
parser.add_argument("--VAL_EVERY", type=int, default=1)
parser.add_argument("--residual_tol_pu", type=float, default=1e-6)
parser.add_argument(
    "--range_step_diagnostic_batches",
    type=int,
    default=4,
    help=(
        "Number of final-test batches used for exact-complex128 K-step voltage "
        "and residual trajectories when global_context_mode=range_post."
    ),
)
parser.add_argument(
    "--report_nr_polish",
    action="store_true",
    help=(
        "After test eval, warm-start a Newton-Raphson solver from the model's "
        "predicted V and count iterations-to-convergence (tol=--nr_polish_tol). "
        "Reports model-warm-start vs flat-start iteration counts and convergence "
        "rate -- quantifies how much solver work the surrogate saves."
    ),
)
parser.add_argument(
    "--nr_polish_solver",
    type=str,
    default="own",
    choices=["own", "pandapower"],
    help=(
        "NR engine for --report_nr_polish. 'own' = your vectorized NR from "
        "ScenarioSynthesis_PPC/newton_raphson_improved.py (the exact solver "
        "used for data generation). 'pandapower' = the pypower-derived polar NR "
        "that pandapower is built on, with pandapower's exact convergence test "
        "(||F||_inf < tol); implemented standalone since pandapower's newtonpf "
        "cannot run without a full net/ppci object."
    ),
)
parser.add_argument("--nr_polish_tol", type=float, default=1e-8,
                    help="Convergence tolerance (pu) for NR-polish (default 1e-8, research-grade).")
parser.add_argument("--nr_polish_max_iter", type=int, default=30,
                    help="Max NR iterations for the polish solve.")
parser.add_argument("--nr_polish_max_cases", type=int, default=200,
                    help="Cap on number of test cases to polish (NR per case is costly).")
parser.add_argument(
    "--report_gradient_alignment_batches",
    type=int,
    default=0,
    help=(
        "On final test evaluation, use this many fixed test mini-batches to "
        "measure unweighted supervised and exact-complex128/local-Ybus "
        "gradient norms, their cosine similarity, and the weighted physics/"
        "supervised norm ratio. Zero disables the diagnostic."
    ),
)
parser.add_argument(
    "--gradient_alignment_split",
    choices=("train", "test"),
    default="train",
    help=(
        "Dataset split used for gradient alignment. The default uses fixed, "
        "non-shuffled training mini-batches so the measured gradients match "
        "the training objective without test leakage."
    ),
)
parser.add_argument(
    "--nr_impl_path",
    type=str,
    default="/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC",
    help="Directory containing newton_raphson_improved.py (for --nr_polish_solver own).",
)
parser.add_argument(
    "--convergence_tol_pu",
    type=float,
    default=1e-3,
    help=(
        "Per-CASE convergence tolerance (pu) for the convergence-rate metric. "
        "A case counts as 'converged' when BOTH max|ΔP| (PV+PQ) and max|ΔQ| "
        "(PQ) fall below this tol -- the same case-level criterion pandapower "
        "uses (||F||_inf < tol). Default 1e-3 pu is an engineering threshold; "
        "set tighter (e.g. 1e-6) to match research-grade NR."
    ),
)
parser.add_argument(
    "--skip_initial_eval",
    action="store_true",
    help="Skip the full train/valid evaluation before epoch 1; useful for large lazy parquet diagnostics",
)

parser.add_argument("--PARQUET", type=str, nargs='+', required=True, help="Path to parquet data file(s)")
parser.add_argument("--seed_value", type=int, default=42)
parser.add_argument(
    "--split_seed",
    type=int,
    default=None,
    help=(
        "Optional dataset-split seed independent of --seed_value. This lets "
        "multi-seed model replicates share exactly the same train/valid/test "
        "examples. By default the model seed is reused for compatibility."
    ),
)
parser.add_argument(
    "--init_checkpoint",
    type=str,
    default="",
    help=(
        "Optional model state_dict used to initialize training. Optimizer and "
        "scheduler state are intentionally reset; this is intended for staged "
        "loss curricula."
    ),
)
parser.add_argument(
    "--no_cache_dense_ybus",
    "--no_cach_dense_ybus",
    action="store_true",
    help="Do not precompute/store dense Ybus in the dataset; reconstruct it inside the model instead",
)
parser.add_argument(
    "--lazy_parquet",
    action="store_true",
    help="Load parquet lazily by row group instead of materializing the full dataset in memory",
)
parser.add_argument(
    "--row_group_cache_size",
    type=int,
    default=2,
    help="How many decoded parquet row groups to keep in RAM when --lazy_parquet is enabled",
)
add_known_operator_args(parser)
add_helm_kol_args(parser)

# NEW
parser.add_argument("--log_to_file", action="store_true", help="Save terminal output to a log file as well")
parser.add_argument("--log_dir", type=str, default="./results/logs", help="Directory for log file")
parser.add_argument(
    "--run_name",
    type=str,
    default="",
    help="Optional short run name for log/checkpoint/plot filenames; defaults to the full configuration name.",
)

args = parser.parse_args()

if min(args.physics_weight, args.mse_weight, args.exact_physics_weight) < 0.0:
    raise ValueError(
        "--physics_weight, --mse_weight, and --exact_physics_weight must be "
        "non-negative"
    )
if args.correction_target_norm_eps <= 0.0:
    raise ValueError("--correction_target_norm_eps must be positive")
if args.exact_physics_weight > 0.0 and args.dataset_complex_dtype != "complex128":
    raise ValueError(
        "--exact_physics_weight requires --dataset_complex_dtype complex128; "
        "casting already-rounded complex64 admittance cannot restore precision"
    )

if args.helm_uniform_path:
    if args.model != "PIGNN_HELM_KOL":
        raise ValueError("--helm_uniform_path requires --model PIGNN_HELM_KOL")
    if "train" in args.mode:
        raise ValueError(
            "--helm_uniform_path is an untrained ablation; use --mode valid_test "
            "rather than a mode containing 'train'."
        )


# ------------------------------------------------------------------
# Effective configuration
# ------------------------------------------------------------------
SEED = args.seed_value
SPLIT_SEED = args.seed_value if args.split_seed is None else args.split_seed

PINN = args.PINN
BLOCK_DIAG = True
NORMALIZE = False
PER_UNIT = True
args.mag_ang_mse = True

MODEL = args.model
BATCH = args.BATCH
EPOCHS = args.EPOCHS
LR = args.LR
VAL_EVERY = args.VAL_EVERY
PARQUET = args.PARQUET
d = args.d
d_hi = args.d_hi
n_heads = args.n_heads
K = args.K
GAMMA = args.gamma
VLIMIT = args.vlimit

torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
class TeeLogger:
    def __init__(self, filename, stream):
        self.stream = stream
        self.log = open(filename, "a", buffering=1)  # line-buffered

    def write(self, message):
        self.stream.write(message)
        self.log.write(message)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass


parquet_filenames = [os.path.splitext(os.path.basename(p))[0] for p in args.PARQUET]
shortened_names = ['_'.join(name.split('_')[:6]) for name in parquet_filenames]
parquet_filename = '_and_'.join(shortened_names)

armijo_tag = "True" if args.use_armijo else "False"
target_sbase_tag = "rawS" if args.target_S_base is None else f"Sbase{args.target_S_base:g}"
start_tag = "flat" if any("manual_flat" in name or "flat" in name for name in parquet_filenames) else "dc"
if any("dc_compile" in name or "_dc_" in name for name in parquet_filenames):
    start_tag = "dc"
loss_mode_tag = "pinn" if args.PINN else "mse"
if args.PINN and args.mse_weight > 0.0:
    loss_mode_tag = f"pinn_mse{args.mse_weight:g}"
elif (not args.PINN) and args.mse_weight > 0.0:
    loss_mode_tag = f"mse_w{args.mse_weight:g}"
vn_tag = "vnfeat" if args.vn_feat else "novn"
grid_cache_tag = (
    ("sharegrid" if args.share_grid else "nosharegrid")
    + ("_sharey" if args.share_ybus else "_nosharey")
    + ("_nocacheY" if args.no_cache_dense_ybus else "_cacheY")
    + ("_lazy" if args.lazy_parquet else "_eager")
)
armijo_detail_tag = "noarmijo"
if args.use_armijo:
    armijo_detail_tag = (
        f"arm{args.armijo_mode}"
        f"_rho{args.armijo_rho:g}"
        f"_bt{args.armijo_max_backtracks}"
        f"_amin{args.armijo_min_alpha:g}"
        f"_anorm{args.armijo_norm}"
    )
stiffness_tag = "rstiff" if args.relative_stiffness_feature else "nostiff"
loss_tag = (
    f"_loss{loss_mode_tag}"
    f"_msew{args.mse_weight:g}"
    f"_physw{args.physics_weight:g}"
    f"_ploss{args.physics_loss_form}"
    f"_pnorm{args.physics_residual_norm}"
    f"_rfeat{args.residual_feature_norm}"
    f"_efeat{args.edge_feature_norm}"
    f"_{stiffness_tag}"
    f"_twhiten{args.correction_target_norm}"
    f"_pfinal{args.physics_final_weight:g}"
    f"_exactphys{args.exact_physics_weight:g}_{args.exact_physics_residual_norm}"
)
run_config_tag = (
    f"_{start_tag}"
    f"_{args.model}"
    f"_{target_sbase_tag}"
    f"_{args.dataset_complex_dtype}"
    f"_{vn_tag}"
    f"_{grid_cache_tag}"
    f"_{armijo_detail_tag}"
    f"_batch{args.BATCH}"
    f"_lr{args.LR:g}"
    f"_seed{args.seed_value}"
    f"_valid{args.valid_ratio:g}"
    f"_{known_operator_tag(args)}"
    f"_{args.solver_update_mode}"
    f"_{args.preconditioner_optimizer}"
    f"_pvm{args.preconditioner_vm_step:g}_pva{args.preconditioner_va_step:g}"
    f"_helmord{args.helm_series_order}_path{args.helm_path_order}"
)


RUNNAME = (
    f"{parquet_filename}_K{args.K}_d{args.d}_dhi{args.d_hi}"
    f"_nheads{args.n_heads}_numattn{args.num_attn_layers}"
    f"_armijo{armijo_tag}{loss_tag}{run_config_tag}"
    f"_ep{args.EPOCHS}_TrainRatio{args.train_ratio:g}"
)
if args.model == "GNSMsg_EdgeSelfAttnKHop":
    RUNNAME = (
        f"{parquet_filename}_solverK{args.K}_khop{args.khop_K}_sigma{args.khop_sigma:g}"
        f"_khop{args.khop_norm}_{args.khop_source}_d{args.d}_dhi{args.d_hi}"
        f"_nheads{args.n_heads}_numattn{args.num_attn_layers}"
        f"_armijo{armijo_tag}{loss_tag}{run_config_tag}"
        f"_ep{args.EPOCHS}_TrainRatio{args.train_ratio:g}"
    )
if args.run_name:
    RUNNAME = args.run_name
os.makedirs(args.ckpt_dir, exist_ok=True)
BEST_CKPT_PATH = os.path.join(args.ckpt_dir, f"{RUNNAME}_{EPOCHS}_best_model.ckpt")
ddp = setup_ddp(args.DDP, timeout_hours=args.ddp_timeout_hours)
atexit.register(cleanup, ddp)

if args.log_to_file and ddp.is_main:
    os.makedirs(args.log_dir, exist_ok=True)
    log_filename = os.path.join(args.log_dir, f"{RUNNAME}_training_log.txt")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = TeeLogger(log_filename, original_stdout)
    sys.stderr = TeeLogger(log_filename, original_stderr)

    def _cleanup_logger():
        stdout_logger = sys.stdout
        stderr_logger = sys.stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        try:
            stdout_logger.flush()
        except Exception:
            pass
        try:
            stderr_logger.flush()
        except Exception:
            pass
        try:
            stdout_logger.close()
        except Exception:
            pass
        try:
            stderr_logger.close()
        except Exception:
            pass

    atexit.register(_cleanup_logger)

    print(f"[logging] stdout/stderr will also be saved to: {log_filename}")

print(
    f"MODEL:{MODEL}, PINN:{PINN}, Block:{BLOCK_DIAG}, d:{d}, d_hi:{d_hi}, n_heads:{n_heads}, "
    f"K:{K}, Runname:{RUNNAME}, PARQUET:{PARQUET}, BATCH:{BATCH}, EP:{EPOCHS}, LR:{LR}, "
    f"model_seed:{SEED}, split_seed:{SPLIT_SEED}, init_checkpoint:{args.init_checkpoint or 'none'}, "
    f"no_cache_dense_ybus:{args.no_cache_dense_ybus}, lazy_parquet:{args.lazy_parquet}, "
    f"row_group_cache_size:{args.row_group_cache_size}, "
    f"physics_weight:{args.physics_weight}, physics_loss_form:{args.physics_loss_form}, physics_residual_norm:{args.physics_residual_norm}, "
    f"residual_feature_norm:{args.residual_feature_norm}, edge_feature_norm:{args.edge_feature_norm}, "
        f"relative_stiffness_feature:{args.relative_stiffness_feature}, "
        f"two_hop_attention_mode:{args.two_hop_attention_mode}, "
        f"global_context_mode:{args.global_context_mode}, "
        f"global_context_gate_mode:{args.global_context_gate_mode}, "
        f"range_num_masters:{args.range_num_masters}, "
        f"range_master_dim:{args.range_master_dim}, "
        f"range_num_heads:{args.range_num_heads}, "
        f"range_positional_encoding:{args.range_positional_encoding}, "
        f"range_pe_dim:{args.range_pe_dim}, "
        f"exact_physics_weight:{args.exact_physics_weight}, exact_physics_residual_norm:{args.exact_physics_residual_norm}, "
    f"correction_target_norm:{args.correction_target_norm}, correction_target_norm_eps:{args.correction_target_norm_eps:g}, "
    f"physics_final_weight:{args.physics_final_weight}, DthetaMax:{args.DthetaMax}, DvmFrac:{args.DvmFrac}, "
    f"khop_K:{args.khop_K}, khop_sigma:{args.khop_sigma}, "
    f"khop_norm:{args.khop_norm}, khop_source:{args.khop_source}, "
    f"dataset_complex_dtype:{args.dataset_complex_dtype}"
    f", known_operator:{known_operator_tag(args)}"
    f", solver_update_mode:{args.solver_update_mode}"
    f", preconditioner_optimizer:{args.preconditioner_optimizer}"
)


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
quiet_non_main(ddp)
device = ddp.device
print("Using device:", device)


# ------------------------------------------------------------------
# Dataset / split
# ------------------------------------------------------------------
full_ds = ChanghunDataset(
    PARQUET,
    per_unit=PER_UNIT,
    device=None,
    no_cache_dense_ybus=args.no_cache_dense_ybus,
    lazy_row_groups=args.lazy_parquet,
    row_group_cache_size=args.row_group_cache_size,
    target_S_base=args.target_S_base,
    share_ybus=args.share_ybus,
    share_grid=args.share_grid,
    complex_dtype=args.dataset_complex_dtype,
)

n_total = len(full_ds)
n_train = int(args.train_ratio * n_total)
n_val = int(args.valid_ratio * n_total)
n_test = n_total - n_train - n_val

train_ds, val_ds, test_ds = random_split(
    full_ds,
    lengths=[n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SPLIT_SEED)
)

def cap_subset(split, max_samples: int):
    if max_samples is None or max_samples <= 0 or len(split) <= max_samples:
        return split
    return Subset(split.dataset, split.indices[:max_samples])

train_ds = cap_subset(train_ds, args.max_train_samples)
val_ds = cap_subset(val_ds, args.max_valid_samples)
test_ds = cap_subset(test_ds, args.max_test_samples)
if args.preload_ram:
    # One decode pass now, then every epoch is pure indexing.
    train_ds = preload_split(train_ds, label='train')
    val_ds = preload_split(val_ds, label='valid')
    if args.preload_test:
        test_ds = preload_split(test_ds, label='test')

n_train, n_val, n_test = len(train_ds), len(val_ds), len(test_ds)

# Only the DDP path sets a DistributedSampler; the other branches leave this
# None (or bind a MultiBucketBatchSampler, which has no set_epoch).
train_sampler = None

if BLOCK_DIAG:
    _lk = dict(batch_size=BATCH, collate_fn=collate_blockdiag)
    train_loader, train_sampler = make_loader(train_ds, ddp, shuffle=True, **_lk)
    val_loader, _ = make_loader(val_ds, ddp, shuffle=False, **_lk)
    # Undistributed: every rank walks the whole test set, so the reported
    # figure is computed exactly as in the single-GPU runs.
    test_loader, _ = make_loader(test_ds, ddp, shuffle=False, distributed=False, **_lk)

else:
    if BATCH == 1:
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    else:
        # Non-blockdiag batching requires homogeneous tensor shapes.
        # With the new parquet metadata that means at least same (N, nl).
        train_signatures = [
            full_ds.get_signature(i) if hasattr(full_ds, "get_signature") else (full_ds[i]["N"], full_ds[i]["nl"])
            for i in train_ds.indices
        ]
        val_signatures = [
            full_ds.get_signature(i) if hasattr(full_ds, "get_signature") else (full_ds[i]["N"], full_ds[i]["nl"])
            for i in val_ds.indices
        ]
        test_signatures = [
            full_ds.get_signature(i) if hasattr(full_ds, "get_signature") else (full_ds[i]["N"], full_ds[i]["nl"])
            for i in test_ds.indices
        ]

        train_sampler = MultiBucketBatchSampler(
            signatures=train_signatures,
            batch_size=BATCH,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = MultiBucketBatchSampler(
            signatures=val_signatures,
            batch_size=BATCH,
            shuffle=False,
            drop_last=True,
        )
        test_sampler = MultiBucketBatchSampler(
            signatures=test_signatures,
            batch_size=BATCH,
            shuffle=False,
            drop_last=True,
        )

        train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
        val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler)
        test_loader  = DataLoader(test_ds,  batch_sampler=test_sampler)

print(f"Dataset sizes | train {n_train}   valid {n_val}   test {n_test}")


def estimate_correction_target_scales(loader, mode, eps):
    """Estimate train-only stds for delta-|V| and wrapped delta-angle.

    This deliberately lives in the training adapter: the network still emits
    physical corrections and the public model architecture is unchanged.
    The current implementation requires a fixed bus ordering, as is the case
    for scenario rows from one PPC grid.
    """
    if mode == "none":
        return None

    sums = sums_sq = counts = reference_vn = None
    with torch.no_grad():
        for batch in loader:
            v0 = batch["V_start"].cpu().double()
            vt = batch["V_newton"].cpu().double()
            dmag = vt[..., 0] - v0[..., 0]
            dang = torch.atan2(
                torch.sin(vt[..., 1] - v0[..., 1]),
                torch.cos(vt[..., 1] - v0[..., 1]),
            )
            delta = torch.stack((dmag, dang), dim=-1)

            if BLOCK_DIAG:
                sizes = [int(x) for x in batch["sizes"].tolist()]
                graphs = list(torch.split(delta.squeeze(0), sizes, dim=0))
                vn_graphs = (
                    list(torch.split(batch["vn_log"].cpu().double().squeeze(0), sizes, dim=0))
                    if "vn_log" in batch else [None] * len(graphs)
                )
            else:
                graphs = list(delta)
                vn_graphs = list(batch["vn_log"].cpu().double()) if "vn_log" in batch else [None] * len(graphs)

            for graph, vn in zip(graphs, vn_graphs):
                n_bus = graph.shape[0]
                if sums is None:
                    sums = torch.zeros(n_bus, 2, dtype=torch.float64)
                    sums_sq = torch.zeros_like(sums)
                    counts = torch.zeros(n_bus, 1, dtype=torch.float64)
                    reference_vn = vn.clone() if vn is not None else None
                if n_bus != sums.shape[0]:
                    raise ValueError(
                        "--correction_target_norm requires one fixed grid/bus ordering; "
                        f"observed {sums.shape[0]} and {n_bus} buses"
                    )
                if mode == "voltage_level":
                    if vn is None:
                        raise ValueError("voltage_level correction normalization requires vn_log in the dataset")
                    if not torch.allclose(vn, reference_vn, atol=1e-6, rtol=0.0):
                        raise ValueError("vn_log/bus ordering changes across training scenarios")
                sums += graph
                sums_sq += graph.square()
                counts += 1.0

    variance = (sums_sq / counts - (sums / counts).square()).clamp_min(0.0)
    if mode == "bus":
        scales = variance.sqrt()
    else:
        # Pool raw first and second moments across every bus/sample belonging
        # to the same nominal-voltage class, rather than averaging bus stds.
        levels = torch.round(reference_vn * 1e6) / 1e6
        scales = torch.empty_like(variance)
        for level in torch.unique(levels):
            mask = levels == level
            group_count = counts[mask].sum()
            group_mean = sums[mask].sum(dim=0) / group_count
            group_var = (sums_sq[mask].sum(dim=0) / group_count - group_mean.square()).clamp_min(0.0)
            scales[mask] = group_var.sqrt()

    scales = scales.clamp_min(float(eps)).float()
    print(
        f"[target-whiten] mode={mode}, buses={scales.shape[0]}, "
        f"sigma_mag min/median/max={scales[:,0].min():.3e}/"
        f"{scales[:,0].median():.3e}/{scales[:,0].max():.3e}, "
        f"sigma_ang min/median/max={scales[:,1].min():.3e}/"
        f"{scales[:,1].median():.3e}/{scales[:,1].max():.3e}"
    )
    return scales


correction_target_scales = estimate_correction_target_scales(
    train_loader, args.correction_target_norm, args.correction_target_norm_eps
)


def correction_scales_for_batch(vpred, n_nodes_per_graph):
    if correction_target_scales is None:
        return None
    scale = correction_target_scales.to(device=vpred.device, dtype=vpred.dtype)
    if n_nodes_per_graph is None:
        if vpred.shape[-2] != scale.shape[0]:
            raise ValueError("correction target scale does not match batch bus count")
        return scale.unsqueeze(0)
    sizes = [int(x) for x in n_nodes_per_graph.tolist()]
    if any(size != scale.shape[0] for size in sizes):
        raise ValueError("correction target scale does not match block-diagonal graph size")
    return scale.repeat(len(sizes), 1).unsqueeze(0)

# ------------------------------------------------------------------
# Model / optimizer / loss
# ------------------------------------------------------------------
if args.model == "GNSMsg":
    model = GNSMsg(
        d=d,
        d_hi=d_hi,
        K=K,
        pinn=PINN,
        gamma=GAMMA,
        v_limit=VLIMIT,
        use_armijo=args.use_armijo
    ).to(device)

elif args.model == "GNSMsg_EdgeSelfAttn":
    model = GNSMsg_EdgeSelfAttn(
        d=d,
        d_hi=d_hi,
        n_heads=n_heads,
        K=K,
        pinn=PINN,
        gamma=GAMMA,
        v_limit=VLIMIT,
        use_armijo=args.use_armijo,
        num_attn_layers=args.num_attn_layers,
        armijo_mode=args.armijo_mode,
        armijo_rho=args.armijo_rho,
        armijo_c1=args.armijo_c1,
        armijo_max_backtracks=args.armijo_max_backtracks,
        armijo_min_alpha=args.armijo_min_alpha,
        armijo_norm=args.armijo_norm,
        dtheta_max=args.DthetaMax,
        dvm_frac=args.DvmFrac,
        physics_loss_form=args.physics_loss_form,
        physics_residual_norm=args.physics_residual_norm,
        residual_feature_norm=args.residual_feature_norm,
        edge_feature_norm=args.edge_feature_norm,
        relative_stiffness_feature=args.relative_stiffness_feature,
        physics_norm_eps=args.physics_norm_eps,
        physics_huber_delta=args.physics_huber_delta,
        physics_final_weight=args.physics_final_weight,
        solver_update_mode=args.solver_update_mode,
        preconditioner_vm_step=args.preconditioner_vm_step,
        preconditioner_va_step=args.preconditioner_va_step,
        preconditioner_log_clip=args.preconditioner_log_clip,
        preconditioner_optimizer=args.preconditioner_optimizer,
        preconditioner_beta1=args.preconditioner_beta1,
        preconditioner_beta2=args.preconditioner_beta2,
        preconditioner_eps=args.preconditioner_eps,
        bus_feat_extra_dim=1 if args.vn_feat else 0,
        bus_type_features=args.bus_type_features,
        heterogeneous_injection_features=args.heterogeneous_injection_features,
        two_hop_attention_mode=args.two_hop_attention_mode,
        global_context_mode=args.global_context_mode,
        global_context_gate_mode=args.global_context_gate_mode,
        range_num_masters=args.range_num_masters,
        range_master_dim=args.range_master_dim,
        range_num_heads=args.range_num_heads,
        range_positional_encoding=args.range_positional_encoding,
        range_pe_dim=args.range_pe_dim,
        range_share_grid=args.share_grid,
    ).to(device)

elif args.model == "GNSMsg_EdgeSelfAttnKHop":
    model = GNSMsg_EdgeSelfAttnKHop(
        d=d,
        d_hi=d_hi,
        n_heads=n_heads,
        K=K,
        pinn=PINN,
        gamma=GAMMA,
        v_limit=VLIMIT,
        use_armijo=args.use_armijo,
        num_attn_layers=args.num_attn_layers,
        armijo_mode=args.armijo_mode,
        armijo_rho=args.armijo_rho,
        armijo_c1=args.armijo_c1,
        armijo_max_backtracks=args.armijo_max_backtracks,
        armijo_min_alpha=args.armijo_min_alpha,
        dtheta_max=args.DthetaMax,
        dvm_frac=args.DvmFrac,
        physics_loss_form=args.physics_loss_form,
        physics_residual_norm=args.physics_residual_norm,
        physics_norm_eps=args.physics_norm_eps,
        physics_huber_delta=args.physics_huber_delta,
        physics_final_weight=args.physics_final_weight,
        khop_K=args.khop_K,
        khop_sigma=args.khop_sigma,
        khop_norm=args.khop_norm,
        khop_source=args.khop_source,
    ).to(device)

elif args.model == "PIGNN_HELM_KOL":
    if args.kol_pf_mode != "off":
        raise ValueError("PIGNN_HELM_KOL cannot be combined with the separate DPF post-operator")
    model = PIGNNHELMKOL(
        d_model=d_hi,
        n_heads=n_heads,
        num_attn_layers=args.num_attn_layers,
        series_order=args.helm_series_order,
        path_order=args.helm_path_order,
        pade_regularization=args.helm_pade_regularization,
        linear_regularization=args.helm_linear_regularization,
        select_best_eval_order=not args.helm_no_best_eval_order,
        physics_loss_form=args.physics_loss_form,
        physics_huber_delta=args.physics_huber_delta,
        uniform_path=args.helm_uniform_path,
    ).to(device)

else:
    raise ValueError(f"Unknown model: {args.model}")

known_operator = build_known_operator(args)
if known_operator is not None:
    known_operator = known_operator.to(device)
    print(f"Known-operator layer: {known_operator}")

def init_weights(model, exclude_modules):
    for module in model.modules():
        if module in exclude_modules:
            continue
        if isinstance(module, nn.Linear):
            if args.weight_init == "sd0.02":
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            elif args.weight_init == "He":
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(args.bias_init)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        else:
            for name, param in module.named_parameters(recurse=False):
                if 'weight' in name and param.dim() > 1:
                    if args.weight_init == "sd0.02":
                        torch.nn.init.normal_(param, mean=0, std=0.02)
                    elif args.weight_init == "He":
                        torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif 'bias' in name:
                    param.data.fill_(args.bias_init)

exclude_modules = []
if args.preserve_zero_heads:
    for attr in (
        "theta_head", "v_head", "m_head", "path_head",
        "theta_preconditioner_head", "v_preconditioner_head",
    ):
        heads = getattr(model, attr, None)
        if heads is not None:
            if isinstance(heads, nn.ModuleList):
                exclude_modules.extend(list(heads.modules()))
            else:
                exclude_modules.extend(list(heads.modules()))
init_weights(model, exclude_modules)

if args.init_checkpoint:
    if not os.path.isfile(args.init_checkpoint):
        raise FileNotFoundError(
            f"--init_checkpoint does not exist: {args.init_checkpoint}"
        )
    initial_state = torch.load(args.init_checkpoint, map_location=device)
    if isinstance(initial_state, dict) and "state_dict" in initial_state:
        initial_state = initial_state["state_dict"]
    incompatible = model.load_state_dict(initial_state, strict=True)
    print(
        f"[init] loaded model weights from {args.init_checkpoint}; "
        f"missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)}; "
        "optimizer/scheduler reset"
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total number of parameters: {count_parameters(model)}")

ACCUM_STEPS = resolve_accum_steps(args, BATCH, ddp.world_size)

if args.lr_scheduler == "CosineAnnealingLR":
    # The scheduler advances once per optimiser step, not once per micro-batch.
    steps_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=args.weight_decay)
    T_0 = args.cosineRestartEpoch * steps_per_epoch
    scheduler = CosineAnnealingWarmRestarts(optim, T_0=T_0, T_mult=1, eta_min=1e-6)
else:
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=args.weight_decay)
    scheduler = None

model = wrap_model(model, ddp,
                   find_unused_parameters=args.ddp_find_unused_parameters,
                   static_graph=args.ddp_static_graph)
accum = GradAccumulator(optim, model, ACCUM_STEPS, max_grad_norm=1.0, scheduler=scheduler)
print(
    f"[optim] micro-batch {BATCH} x world {ddp.world_size} "
    f"x grad_accum_steps {ACCUM_STEPS} "
    f"= effective batch {BATCH * ddp.world_size * ACCUM_STEPS}"
)


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    return dtype


def build_dense_y_from_branchrows_single(
    N,
    Branch_f_bus,
    Branch_t_bus,
    Branch_status,
    Branch_tau,
    Branch_shift_deg,
    Branch_y_series_from,
    Branch_y_series_to,
    Branch_y_series_ft,
    Branch_y_shunt_from,
    Branch_y_shunt_to,
    Y_shunt_bus,
):
    device = Branch_f_bus.device
    dtype = Branch_y_series_ft.dtype

    Y = torch.zeros(N, N, dtype=dtype, device=device)
    Y.diagonal().add_(Y_shunt_bus.to(dtype))

    mask = (Branch_status != 0)
    if mask.sum() == 0:
        return Y

    f = Branch_f_bus[mask].long()
    t = Branch_t_bus[mask].long()

    real_dtype = _real_dtype(dtype)
    tau = Branch_tau[mask].to(real_dtype)
    theta = torch.deg2rad(Branch_shift_deg[mask].to(real_dtype))
    a = tau.to(dtype) * torch.exp(1j * theta.to(dtype))

    y_from = Branch_y_series_from[mask].to(dtype)
    y_to = Branch_y_series_to[mask].to(dtype)
    ysh_f = Branch_y_shunt_from[mask].to(dtype)
    ysh_t = Branch_y_shunt_to[mask].to(dtype)

    Yff = (y_from + ysh_f / 2.0) / (a * torch.conj(a))
    Ytt = (y_to + ysh_t / 2.0)
    Yft = -y_from / torch.conj(a)
    Ytf = -y_to / a

    Y.index_put_((f, f), Yff, accumulate=True)
    Y.index_put_((t, t), Ytt, accumulate=True)
    Y.index_put_((f, t), Yft, accumulate=True)
    Y.index_put_((t, f), Ytf, accumulate=True)

    return Y


def ensure_dense_y_for_metrics(
    Y,
    bus_type,
    Branch_f_bus,
    Branch_t_bus,
    Branch_status,
    Branch_tau,
    Branch_shift_deg,
    Branch_y_series_from,
    Branch_y_series_to,
    Branch_y_series_ft,
    Branch_y_shunt_from,
    Branch_y_shunt_to,
    Y_shunt_bus,
):
    if Y is not None:
        if Y.is_sparse:
            return Y
        return Y.unsqueeze(0) if Y.dim() == 2 else Y

    if Y_shunt_bus is None:
        raise ValueError("Y is None and Y_shunt_bus is None; cannot reconstruct Y for residual metrics.")

    B, N = bus_type.shape
    if B == 1:
        return build_dense_y_from_branchrows_single(
            N,
            Branch_f_bus.squeeze(0),
            Branch_t_bus.squeeze(0),
            Branch_status.squeeze(0),
            Branch_tau.squeeze(0),
            Branch_shift_deg.squeeze(0),
            Branch_y_series_from.squeeze(0),
            Branch_y_series_to.squeeze(0),
            Branch_y_series_ft.squeeze(0),
            Branch_y_shunt_from.squeeze(0),
            Branch_y_shunt_to.squeeze(0),
            Y_shunt_bus.squeeze(0),
        ).unsqueeze(0)

    Ys = []
    for b in range(B):
        Ys.append(build_dense_y_from_branchrows_single(
            N,
            Branch_f_bus[b],
            Branch_t_bus[b],
            Branch_status[b],
            Branch_tau[b],
            Branch_shift_deg[b],
            Branch_y_series_from[b],
            Branch_y_series_to[b],
            Branch_y_series_ft[b],
            Branch_y_shunt_from[b],
            Branch_y_shunt_to[b],
            Y_shunt_bus[b],
        ))
    return torch.stack(Ys, dim=0)


def _per_bus_view(V, n_nodes_per_graph):
    """Reshape a batch of reference voltages to [n_graphs, n_buses, 2].

    Returns None when the batch packs graphs of differing size, in which case
    "the same bus" is not defined across the batch and the per-bus baseline
    does not apply.
    """
    if n_nodes_per_graph is None:
        return V if V.dim() == 3 else None
    sizes = n_nodes_per_graph
    n = int(sizes[0].item())
    if int(sizes.min().item()) != n or int(sizes.max().item()) != n:
        return None
    flat = V.reshape(-1, V.shape[-1])
    if flat.shape[0] % n:
        return None
    return flat.reshape(-1, n, V.shape[-1])


def compute_power_flow_residual_metrics(Y, Vpred, Sset, bus_type, *, n_nodes_per_graph=None, S_base=None):
    if Y.dim() == 2 and not Y.is_sparse:
        Y = Y.unsqueeze(0)

    if Y.is_complex():
        complex_dtype = Y.dtype
        real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    else:
        real_dtype = Y.dtype
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

    v = Vpred[..., 0].to(dtype=real_dtype)
    th = Vpred[..., 1].to(dtype=real_dtype)
    Sset = Sset.to(device=Y.device, dtype=complex_dtype)

    Vc = v * torch.exp(1j * th)
    Ic = ybus_matvec(Y, Vc)
    Sc = Vc * Ic.conj()

    P_set, Q_set = Sset.real, Sset.imag
    slack_mask = (bus_type == 1)
    pv_mask = (bus_type == 2)

    dp_abs = (P_set - Sc.real).abs()
    dq_abs = (Q_set - Sc.imag).abs()
    p_mask = ~slack_mask
    q_mask = ~(slack_mask | pv_mask)

    if n_nodes_per_graph is not None:
        max_dp_pu = []
        max_dq_pu = []
        offset = 0
        for size in n_nodes_per_graph.tolist():
            size = int(size)
            sl = slice(offset, offset + size)

            dp_g = dp_abs[0, sl]
            dq_g = dq_abs[0, sl]
            p_mask_g = p_mask[0, sl]
            q_mask_g = q_mask[0, sl]

            if p_mask_g.any():
                max_dp_pu.append(dp_g[p_mask_g].max())
            else:
                max_dp_pu.append(dp_g.new_zeros(()))

            if q_mask_g.any():
                max_dq_pu.append(dq_g[q_mask_g].max())
            else:
                max_dq_pu.append(dq_g.new_zeros(()))

            offset += size

        max_dp_pu = torch.stack(max_dp_pu)
        max_dq_pu = torch.stack(max_dq_pu)
    else:
        max_dp_pu = dp_abs.masked_fill(~p_mask, 0.0).amax(dim=-1)
        max_dq_pu = dq_abs.masked_fill(~q_mask, 0.0).amax(dim=-1)
        max_dq_pu = torch.where(q_mask.any(dim=-1), max_dq_pu, torch.zeros_like(max_dq_pu))

    metrics = {
        "max_dp_pu": max_dp_pu,
        "max_dq_pu": max_dq_pu,
        "dp_abs_valid": dp_abs[p_mask].detach(),
        "dq_abs_valid": dq_abs[q_mask].detach(),
    }

    if S_base is not None:
        S_base = S_base.to(max_dp_pu.device, dtype=max_dp_pu.dtype).reshape(-1)
        if S_base.numel() == 1 and max_dp_pu.numel() != 1:
            S_base = S_base.expand(max_dp_pu.numel())
        # S_base is stored in VA. A per-unit power residual multiplied by
        # S_base is VA; divide by 1e6 before labelling it MW/MVAr.
        metrics["max_dp_mva"] = max_dp_pu * S_base / 1e6
        metrics["max_dq_mva"] = max_dq_pu * S_base / 1e6

    return metrics


def format_residual_summary(max_dp_pu, max_dq_pu):
    return f"(ΔP∞ {max_dp_pu:.3e} pu, ΔQ∞ {max_dq_pu:.3e} pu)"


# Kept in step with REPORT_TOLERANCES in train_valid_test_gridfm.py, which the
# other three drivers share. Duplicated rather than imported on purpose: that
# module pulls in torch_geometric and gridfm_graphkit, and this driver needs
# neither. Two constants are cheaper to keep in sync than a heavy dependency.
REPORT_TOLERANCES = (1e-2, 1e-3)


def _tol_key(channel, tol):
    return f"frac_{channel}_le_{tol:.0e}"


def _empty_residual_distribution():
    return {
        "max_dp_pu": 0.0,
        "max_dq_pu": 0.0,
        "mean_dp_pu": 0.0,
        "mean_dq_pu": 0.0,
        "median_dp_pu": 0.0,
        "median_dq_pu": 0.0,
        "p95_dp_pu": 0.0,
        "p95_dq_pu": 0.0,
        "p99_dp_pu": 0.0,
        "p99_dq_pu": 0.0,
        "rmse_dp_pu": 0.0,
        "rmse_dq_pu": 0.0,
        "frac_dp_below_tol": 0.0,
        "frac_dq_below_tol": 0.0,
        **{_tol_key(c, t): 0.0 for c in ("dp", "dq") for t in REPORT_TOLERANCES},
        "n_dp": 0,
        "n_dq": 0,
        "convergence_rate": 0.0,
        "convergence_tol_pu": 0.0,
        "n_converged": 0,
        "n_cases": 0,
    }


def _safe_quantile(t: torch.Tensor, q: float) -> float:
    """Compute quantile robustly regardless of tensor size.

    torch.quantile has a hard element limit (~16M). For larger tensors
    (e.g. 28800 samples × 722 buses = 20.8M elements) we fall back to
    numpy which has no such restriction. numpy.percentile on CPU is fast
    enough for end-of-epoch metric computation.
    """
    try:
        return torch.quantile(t, q).item()
    except RuntimeError:
        # numpy fallback — move to CPU first if needed
        arr = t.cpu().numpy() if t.is_cuda else t.numpy()
        return float(np.percentile(arr, q * 100))


def finalize_residual_distribution(dp_values, dq_values, *, tol_pu):
    dist = _empty_residual_distribution()

    if dp_values:
        dp = torch.cat(dp_values).float()
        dist.update({
            "max_dp_pu":           dp.max().item(),
            "mean_dp_pu":          dp.mean().item(),
            "median_dp_pu":        _safe_quantile(dp, 0.50),
            "p95_dp_pu":           _safe_quantile(dp, 0.95),
            "p99_dp_pu":           _safe_quantile(dp, 0.99),
            "rmse_dp_pu":          torch.sqrt((dp ** 2).mean()).item(),
            "frac_dp_below_tol":   (dp <= tol_pu).float().mean().item(),
            **{_tol_key("dp", t): (dp <= t).float().mean().item()
               for t in REPORT_TOLERANCES},
            "n_dp":                int(dp.numel()),
        })

    if dq_values:
        dq = torch.cat(dq_values).float()
        dist.update({
            "max_dq_pu":           dq.max().item(),
            "mean_dq_pu":          dq.mean().item(),
            "median_dq_pu":        _safe_quantile(dq, 0.50),
            "p95_dq_pu":           _safe_quantile(dq, 0.95),
            "p99_dq_pu":           _safe_quantile(dq, 0.99),
            "rmse_dq_pu":          torch.sqrt((dq ** 2).mean()).item(),
            "frac_dq_below_tol":   (dq <= tol_pu).float().mean().item(),
            **{_tol_key("dq", t): (dq <= t).float().mean().item()
               for t in REPORT_TOLERANCES},
            "n_dq":                int(dq.numel()),
        })

    return dist


def format_residual_distribution_compact(dist):
    text = (
        f"(mean |ΔP| {dist['mean_dp_pu']:.3e}, |ΔQ| {dist['mean_dq_pu']:.3e} pu; "
        f"p95 |ΔP| {dist['p95_dp_pu']:.3e}, |ΔQ| {dist['p95_dq_pu']:.3e} pu; "
        f"tol≤ {dist['frac_dp_below_tol']:.2%} P, {dist['frac_dq_below_tol']:.2%} Q; "
        f"conv {dist.get('convergence_rate', 0.0):.2%}@{dist.get('convergence_tol_pu', 0.0):.0e}; "
        + "; ".join(
            f"≤{t:.0e} {dist.get(_tol_key('dp', t), 0.0):.2%} P, "
            f"{dist.get(_tol_key('dq', t), 0.0):.2%} Q"
            for t in REPORT_TOLERANCES
        )
        + ")"
    )
    if "vm_r2" in dist:
        text += f" R2 |V| {dist['vm_r2']:+.3f}, theta {dist['va_r2']:+.3f}"
    if "kol_certification_rate" in dist:
        text += (
            f" KOL F∞ {dist['kol_initial_mean']:.3e}->{dist['kol_final_mean']:.3e}; "
            f"cert {dist['kol_certification_rate']:.2%}@{dist['kol_tolerance']:.0e}; "
            f"steps {dist['kol_mean_iterations']:.1f}"
        )
    if "helm_mean_selected_order" in dist:
        text += (
            f" HELM F∞ mean/max {dist['helm_mean_max_mismatch']:.3e}/"
            f"{dist['helm_max_mismatch']:.3e}; order {dist['helm_mean_selected_order']:.1f}"
        )
    if "global_gate_mean" in dist:
        text += (
            f" global_gate {dist['global_gate_mean']:+.3e}"
            f" entropy {dist['global_attention_entropy_mean']:.3f}"
        )
        if "global_inner_gate_mean" in dist:
            text += (
                f" sigmoid_mean {dist['global_inner_gate_mean']:.3f}"
                f" sigmoid<.1 {dist['global_inner_gate_low_fraction']:.2%}"
            )
    if "range_effective_masters_mean" in dist:
        text += (
            f" RANGE aggH {dist['range_aggregation_entropy_mean']:.3f}"
            f" bcastH {dist['range_broadcast_entropy_mean']:.3f}"
            f" effM {dist['range_effective_masters_mean']:.2f}"
            f" attnCos {dist['range_master_attention_cosine_mean']:+.3f}"
            f" embCos {dist['range_master_embedding_similarity_mean']:+.3f}"
            f" self {dist['range_self_loop_fraction_mean']:.2%}"
        )
    return text


def format_residual_distribution_full(dist, *, tol_pu):
    text = (
        f"Residual distribution over PV+PQ/PQ buses (tol={tol_pu:.1e} pu):\n"
        f"  max    |ΔP| {dist['max_dp_pu']:.4e} pu | |ΔQ| {dist['max_dq_pu']:.4e} pu\n"
        f"  mean   |ΔP| {dist['mean_dp_pu']:.4e} pu | |ΔQ| {dist['mean_dq_pu']:.4e} pu\n"
        f"  median |ΔP| {dist['median_dp_pu']:.4e} pu | |ΔQ| {dist['median_dq_pu']:.4e} pu\n"
        f"  p95    |ΔP| {dist['p95_dp_pu']:.4e} pu | |ΔQ| {dist['p95_dq_pu']:.4e} pu\n"
        f"  p99    |ΔP| {dist['p99_dp_pu']:.4e} pu | |ΔQ| {dist['p99_dq_pu']:.4e} pu\n"
        f"  RMSE   ΔP   {dist['rmse_dp_pu']:.4e} pu | ΔQ   {dist['rmse_dq_pu']:.4e} pu\n"
        f"  frac below tol: P {dist['frac_dp_below_tol']:.2%} ({dist['n_dp']} entries), "
        f"Q {dist['frac_dq_below_tol']:.2%} ({dist['n_dq']} entries)\n"
        + "".join(
            f"  frac ≤ {t:.0e} pu: P {dist.get(_tol_key('dp', t), 0.0):.2%}, "
            f"Q {dist.get(_tol_key('dq', t), 0.0):.2%}\n"
            for t in REPORT_TOLERANCES
        )
        + f"  convergence rate (per-case, max|ΔP|&max|ΔQ| < {dist.get('convergence_tol_pu',0.0):.1e} pu): "
        f"{dist.get('convergence_rate',0.0):.2%} "
        f"({dist.get('n_converged',0)}/{dist.get('n_cases',0)} cases)"
    )
    if "vm_r2" in dist:
        text += (
            f"\n  voltage R2 ({dist.get('r2_baseline','per_bus')} baseline): "
            f"|V| {dist['vm_r2']:+.6f}; "
            f"theta {dist['va_r2']:+.6f}"
        )
    if "vm_r2_global" in dist:
        # Kept for continuity with logs written before the per-bus baseline
        # existed; it is the weaker test and should not be quoted on its own.
        text += (
            f"\n  voltage R2 (global-mean baseline, weaker): "
            f"|V| {dist['vm_r2_global']:+.6f}; "
            f"theta {dist['va_r2_global']:+.6f}"
        )
    if "kol_certification_rate" in dist:
        text += (
            f"\n  KOL correction: mean F_inf {dist['kol_initial_mean']:.4e} -> "
            f"{dist['kol_final_mean']:.4e} pu; certified "
            f"{dist['kol_certification_rate']:.2%} at {dist['kol_tolerance']:.1e} pu; "
            f"mean steps {dist['kol_mean_iterations']:.2f}"
        )
    if "helm_mean_selected_order" in dist:
        text += (
            f"\n  HELM KOL: mean/max F_inf {dist['helm_mean_max_mismatch']:.4e}/"
            f"{dist['helm_max_mismatch']:.4e} pu; mean selected Pade order "
            f"{dist['helm_mean_selected_order']:.2f}"
        )
    if "global_gate_mean" in dist:
        text += (
            f"\n  global context: gate {dist['global_gate_mean']:+.4e}; "
            f"normalized attention entropy {dist['global_attention_entropy_mean']:.4f}"
        )
        if "global_inner_gate_mean" in dist:
            text += (
                f"; sigmoid mean {dist['global_inner_gate_mean']:.4f}; "
                f"fraction below 0.1 {dist['global_inner_gate_low_fraction']:.2%}"
            )
    if "range_effective_masters_mean" in dist:
        text += (
            f"\n  RANGE context: aggregation/broadcast normalized entropy "
            f"{dist['range_aggregation_entropy_mean']:.4f}/"
            f"{dist['range_broadcast_entropy_mean']:.4f}; effective masters "
            f"{dist['range_effective_masters_mean']:.3f}; master-attention cosine "
            f"{dist['range_master_attention_cosine_mean']:+.4f}; master-embedding "
            f"cosine {dist['range_master_embedding_similarity_mean']:+.4f}; "
            f"self-loop mass {dist['range_self_loop_fraction_mean']:.2%}; "
            f"mean token utilization {dist['range_token_utilization_mean']}"
        )
        if "range_step_diagnostics" in dist:
            text += (
                f"\n  RANGE K-step trajectory "
                f"({dist.get('range_step_diagnostic_cases', 0)} cases, "
                "exact-complex128 residuals):"
            )
            for step in dist["range_step_diagnostics"]:
                text += (
                    f"\n    k={step['k']:02d}: |V| {step['vm_rmse']:.4e}, "
                    f"theta {step['va_rmse_deg']:.4e} deg; "
                    f"P med/p95/rms/max {step['median_dp']:.3e}/"
                    f"{step['p95_dp']:.3e}/{step['rmse_dp']:.3e}/"
                    f"{step['max_dp']:.3e}; Q {step['median_dq']:.3e}/"
                    f"{step['p95_dq']:.3e}/{step['rmse_dq']:.3e}/"
                    f"{step['max_dq']:.3e}"
                )
    if "armijo_acceptance_rate" in dist:
        text += (
            f"\n  Armijo: strict accept {dist['armijo_acceptance_rate']:.2%}; "
            f"forced-safe {dist['armijo_forced_rate']:.2%}; "
            f"discard/reject {dist['armijo_reject_rate']:.2%}; "
            f"mean alpha {dist['armijo_mean_alpha']:.4f}"
        )
    return text


# ------------------------------------------------------------------
# Epoch runner
# ------------------------------------------------------------------
def _gradient_group_names(parameter_name):
    groups = ["global"]
    head_prefixes = ("theta_head.", "v_head.", "m_head.")
    if parameter_name.startswith(head_prefixes):
        groups.append("all_heads")
        family, step, _ = parameter_name.split(".", 2)
        groups.extend((family, f"step_{int(step):02d}"))
    else:
        groups.append("backbone")
    return groups


def _record_gradient_alignment(accumulator, supervised_loss, physics_loss):
    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]
    grad_sup = torch.autograd.grad(
        supervised_loss, params, retain_graph=True, allow_unused=True
    )
    grad_phys = torch.autograd.grad(
        physics_loss, params, retain_graph=True, allow_unused=True
    )
    grouped = {}
    for (name, _), gs, gp in zip(named_params, grad_sup, grad_phys):
        if gs is None and gp is None:
            continue
        for group in _gradient_group_names(name):
            sums = grouped.setdefault(group, [0.0, 0.0, 0.0])
            if gs is not None:
                gsd = gs.detach().double()
                sums[0] += float(torch.sum(gsd * gsd).item())
            if gp is not None:
                gpd = gp.detach().double()
                sums[1] += float(torch.sum(gpd * gpd).item())
            if gs is not None and gp is not None:
                sums[2] += float(torch.sum(gsd * gpd).item())
    for group, (sup_sq, phys_sq, dot) in grouped.items():
        sup_norm = math.sqrt(max(sup_sq, 0.0))
        phys_norm = math.sqrt(max(phys_sq, 0.0))
        denom = sup_norm * phys_norm
        cosine = dot / denom if denom > 0.0 else float("nan")
        weighted_ratio = (
            args.exact_physics_weight * phys_norm / sup_norm
            if sup_norm > 0.0 else float("nan")
        )
        accumulator.setdefault(group, []).append(
            (sup_norm, phys_norm, cosine, weighted_ratio)
        )


def _print_gradient_alignment(accumulator, supervised_losses, physics_losses):
    if not accumulator:
        return
    print(
        f"[grad-align] batches={len(supervised_losses)} "
        f"exact_weight={args.exact_physics_weight:.6g} "
        f"supervised_loss_mean={np.mean(supervised_losses):.6e} "
        f"exact_local_loss_mean={np.mean(physics_losses):.6e}"
    )
    for group in sorted(accumulator, key=lambda x: (x != "global", x)):
        values = np.asarray(accumulator[group], dtype=np.float64)
        means = np.nanmean(values, axis=0)
        stds = np.nanstd(values, axis=0, ddof=1) if len(values) > 1 else np.zeros(4)
        print(
            f"[grad-align] group={group} "
            f"sup_norm={means[0]:.6e}+/-{stds[0]:.2e} "
            f"phys_norm={means[1]:.6e}+/-{stds[1]:.2e} "
            f"cosine={means[2]:+.6f}+/-{stds[2]:.2e} "
            f"weighted_phys_over_sup={means[3]:.6e}+/-{stds[3]:.2e}"
        )


def run_epoch(
    loader,
    *,
    train: bool,
    pinn: bool,
    diagnose_gradients: bool = False,
    diagnostic_only: bool = False,
    record_step_diagnostics: bool = False,
):
    model.train() if train else model.eval()

    sum_loss = 0.0
    sum_mse = 0.0
    sum_mse_mag = 0.0
    sum_mse_ang = 0.0
    sum_max_dp_pu = 0.0
    sum_max_dq_pu = 0.0
    sum_max_dp_mva = 0.0
    sum_max_dq_mva = 0.0
    sum_vm_sse = 0.0
    sum_va_sse = 0.0
    sum_vm_target = 0.0
    sum_va_target = 0.0
    sum_vm_target_sq = 0.0
    sum_va_target_sq = 0.0
    r2_value_count = 0
    # Per-bus R^2 baseline.  The pooled-global variance below answers "is the
    # model better than one number for the whole grid", which on a real network
    # is dominated by the static bus-to-bus difference in nominal voltage level
    # and is therefore easy to beat without learning anything about the
    # scenario.  These buffers instead accumulate each bus's own statistics, so
    # the baseline becomes "predict each bus's usual voltage", matching the
    # sigma used by the report tables.  Only defined when every graph in the
    # run has the same bus count; per_bus_ok goes False otherwise.
    per_bus_vm_sum = None
    per_bus_sin_sum = None
    per_bus_cos_sum = None
    per_bus_n_graphs = 0
    per_bus_ok = True
    dp_dist_values = []
    dq_dist_values = []
    n_graphs_total = 0
    n_converged = 0   # per-case: both max|ΔP| and max|ΔQ| < convergence_tol_pu
    kol_initial_values = []
    kol_final_values = []
    kol_converged_values = []
    kol_iteration_values = []
    helm_residual_values = []
    helm_order_values = []
    gradient_alignment = {}
    gradient_supervised_losses = []
    gradient_physics_losses = []
    global_gate_values = []
    global_entropy_values = []
    global_inner_gate_values = []
    global_inner_gate_low_values = []
    range_diagnostics_values = []
    armijo_status_counts = {
        "accept": 0, "fallback": 0, "safe": 0, "discard": 0, "reject": 0
    }
    armijo_alpha_values = []
    range_step_dp_values = [[] for _ in range(K)]
    range_step_dq_values = [[] for _ in range(K)]
    range_step_vm_sse = np.zeros(K, dtype=np.float64)
    range_step_va_sse = np.zeros(K, dtype=np.float64)
    range_step_value_count = np.zeros(K, dtype=np.int64)
    range_step_batches_done = 0
    range_step_cases = 0

    diagnostic_model = unwrap(model)
    diagnostic_model.record_step_states = bool(
        record_step_diagnostics and args.range_step_diagnostic_batches > 0
    )

    if train:
        accum.zero()

    with torch.set_grad_enabled(train or diagnose_gradients):
        for batch in loader:
            diagnostic_model.record_step_states = bool(
                record_step_diagnostics
                and range_step_batches_done < args.range_step_diagnostic_batches
            )
            if BLOCK_DIAG and "sizes" in batch:
                B_eff = int(batch["sizes"].numel())
            else:
                B_eff = int(batch["bus_type"].size(0))
            n_graphs_total += B_eff

            if BLOCK_DIAG:
                n_nodes_per_graph = batch["sizes"].to(device)
            else:
                n_nodes_per_graph = None

            bus_type = batch["bus_type"].to(device)

            Branch_f_bus = batch["Branch_f_bus"].to(device)
            Branch_t_bus = batch["Branch_t_bus"].to(device)
            Branch_status = batch["Branch_status"].to(device)
            Branch_tau = batch["Branch_tau"].to(device)
            Branch_shift_deg = batch["Branch_shift_deg"].to(device)

            Branch_y_series_from = batch["Branch_y_series_from"].to(device)
            Branch_y_series_to   = batch["Branch_y_series_to"].to(device)
            Branch_y_series_ft   = batch["Branch_y_series_ft"].to(device)

            Branch_y_shunt_from = batch["Branch_y_shunt_from"].to(device)
            Branch_y_shunt_to   = batch["Branch_y_shunt_to"].to(device)

            Is_trafo = batch["Is_trafo"].to(device)
            Y_shunt_bus = batch["Y_shunt_bus"].to(device)

            Y = batch.get("Ybus", None)
            if Y is not None:
                Y = Y.to(device)
            S_base = batch.get("S_base", None)
            if S_base is not None:
                S_base = S_base.to(device)

            Sstart = batch["S_start"].to(device)
            Ustart = batch["U_start"].to(device)
            Vstart = batch["V_start"].to(device)
            Vnewton = batch["V_newton"].to(device)
            vn_log = batch["vn_log"].to(device) if "vn_log" in batch else None

            Y_metric = Y
            Sstart_metric = Sstart
            S_base_metric = S_base
            Branch_tau_metric = Branch_tau
            Branch_shift_deg_metric = Branch_shift_deg
            Branch_y_series_from_metric = Branch_y_series_from
            Branch_y_series_to_metric = Branch_y_series_to
            Branch_y_series_ft_metric = Branch_y_series_ft
            Branch_y_shunt_from_metric = Branch_y_shunt_from
            Branch_y_shunt_to_metric = Branch_y_shunt_to
            Y_shunt_bus_metric = Y_shunt_bus

            # Keep the neural network on its normal float32/complex64 path.
            # The dataset can decode LVN admittances in complex128 so that
            # validation/test residual diagnostics avoid complex64 cancellation
            # artifacts, but PyTorch Linear layers here are float32.
            if args.dataset_complex_dtype == "complex128":
                Branch_tau = Branch_tau.float()
                Branch_shift_deg = Branch_shift_deg.float()
                Branch_y_series_from = Branch_y_series_from.to(torch.complex64)
                Branch_y_series_to = Branch_y_series_to.to(torch.complex64)
                Branch_y_series_ft = Branch_y_series_ft.to(torch.complex64)
                Branch_y_shunt_from = Branch_y_shunt_from.to(torch.complex64)
                Branch_y_shunt_to = Branch_y_shunt_to.to(torch.complex64)
                Y_shunt_bus = Y_shunt_bus.to(torch.complex64)
                if args.model != "PIGNN_HELM_KOL":
                    if Y is not None:
                        Y = Y.to(torch.complex64)
                    Sstart = Sstart.to(torch.complex64)
                    Ustart = Ustart.to(torch.complex64)
                    Vstart = Vstart.float()
                    Vnewton = Vnewton.float()
                if vn_log is not None:
                    vn_log = vn_log.float()

            Y_exact = ensure_dense_y_for_metrics(
                Y_metric,
                bus_type,
                Branch_f_bus,
                Branch_t_bus,
                Branch_status,
                Branch_tau_metric,
                Branch_shift_deg_metric,
                Branch_y_series_from_metric,
                Branch_y_series_to_metric,
                Branch_y_series_ft_metric,
                Branch_y_shunt_from_metric,
                Branch_y_shunt_to_metric,
                Y_shunt_bus_metric,
            )
            kol_diag = None

            if pinn:
                Vraw, loss_phys = model(
                    bus_type,
                    Branch_f_bus, Branch_t_bus, Branch_status,
                    Branch_tau, Branch_shift_deg,
                    Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
                    Branch_y_shunt_from, Branch_y_shunt_to,
                    Is_trafo,
                    Y,
                    Sstart,
                    Vstart,
                    n_nodes_per_graph=n_nodes_per_graph,
                    Y_shunt_bus=Y_shunt_bus,
                    vn_log=vn_log,
                )

                if known_operator is not None:
                    Vpred, kol_diag = known_operator(
                        Vraw,
                        Y_exact,
                        Sstart_metric,
                        bus_type,
                        V_fixed=batch["V_start"].to(device),
                        sizes=n_nodes_per_graph,
                        differentiable=train,
                    )
                    loss_phys = loss_phys + args.kol_final_physics_weight * power_flow_residual_loss(
                        Y_exact,
                        Vpred,
                        Sstart_metric,
                        bus_type,
                        sizes=n_nodes_per_graph,
                    )
                else:
                    Vpred = Vraw

                dmag = (Vpred[..., 0] - Vnewton[..., 0])
                dang = torch.atan2(
                    torch.sin(Vpred[..., 1] - Vnewton[..., 1]),
                    torch.cos(Vpred[..., 1] - Vnewton[..., 1])
                )
                mse_mag = torch.mean(dmag ** 2)
                mse_ang = torch.mean(dang ** 2)
                mse = mse_mag + mse_ang
                target_scale = correction_scales_for_batch(Vpred, n_nodes_per_graph)
                if target_scale is None:
                    supervised_mse = mse
                else:
                    supervised_mse = torch.mean((dmag / target_scale[..., 0]) ** 2) + torch.mean(
                        (dang / target_scale[..., 1]) ** 2
                    )
                # Combined loss: L = w_phys*L_phys + w_mse*L_MSE.
                # mse_weight=0 (default) → pure PINN (original behaviour)
                # mse_weight>0 → MSE anchor pulls V toward V_newton while PINN
                #        satisfies physics; stabilizes training on stiff/
                #        multi-voltage grids where pure PINN converges to
                #        a non-V_newton physics-feasible solution.
                if args.mse_weight > 0.0:
                    loss = args.physics_weight * loss_phys + args.mse_weight * supervised_mse
                else:
                    loss = args.physics_weight * loss_phys
                if known_operator is not None and args.kol_raw_loss_weight > 0.0:
                    raw_dmag = Vraw[..., 0] - Vnewton[..., 0]
                    raw_dang = torch.atan2(
                        torch.sin(Vraw[..., 1] - Vnewton[..., 1]),
                        torch.cos(Vraw[..., 1] - Vnewton[..., 1]),
                    )
                    raw_mse = torch.mean(raw_dmag ** 2) + torch.mean(raw_dang ** 2)
                    loss = loss + args.kol_raw_loss_weight * raw_mse

                if train and not loss.requires_grad:
                    p0 = next(model.parameters())
                    loss = loss + 0.0 * p0.norm()
                    print("[warn] physics loss detached for this batch; applied zero-grad guard.")
            else:
                Vraw = model(
                    bus_type,
                    Branch_f_bus, Branch_t_bus, Branch_status,
                    Branch_tau, Branch_shift_deg,
                    Branch_y_series_from, Branch_y_series_to, Branch_y_series_ft,
                    Branch_y_shunt_from, Branch_y_shunt_to,
                    Is_trafo,
                    Y,
                    Sstart,
                    Vstart,
                    n_nodes_per_graph=n_nodes_per_graph,
                    Y_shunt_bus=Y_shunt_bus,
                    vn_log=vn_log,
                )

                if known_operator is not None:
                    Vpred, kol_diag = known_operator(
                        Vraw,
                        Y_exact,
                        Sstart_metric,
                        bus_type,
                        V_fixed=batch["V_start"].to(device),
                        sizes=n_nodes_per_graph,
                        differentiable=train,
                    )
                else:
                    Vpred = Vraw

                dmag = (Vpred[..., 0] - Vnewton[..., 0])
                dang = torch.atan2(
                    torch.sin(Vpred[..., 1] - Vnewton[..., 1]),
                    torch.cos(Vpred[..., 1] - Vnewton[..., 1])
                )
                mse_mag = torch.mean(dmag ** 2)
                mse_ang = torch.mean(dang ** 2)
                mse = mse_mag + mse_ang
                target_scale = correction_scales_for_batch(Vpred, n_nodes_per_graph)
                if target_scale is None:
                    supervised_mse = mse
                else:
                    supervised_mse = torch.mean((dmag / target_scale[..., 0]) ** 2) + torch.mean(
                        (dang / target_scale[..., 1]) ** 2
                    )
                loss = supervised_mse
                if known_operator is not None and args.kol_raw_loss_weight > 0.0:
                    raw_dmag = Vraw[..., 0] - Vnewton[..., 0]
                    raw_dang = torch.atan2(
                        torch.sin(Vraw[..., 1] - Vnewton[..., 1]),
                        torch.cos(Vraw[..., 1] - Vnewton[..., 1]),
                    )
                    raw_mse = torch.mean(raw_dmag ** 2) + torch.mean(raw_dang ** 2)
                    loss = loss + args.kol_raw_loss_weight * raw_mse

                # Zero-grad guard for pure-MSE mode (pinn=False).
                # Root cause: when Armijo rejects all K steps for every
                # iteration, Vpred == V_start (unchanged), which is a
                # constant with no grad_fn. The MSE against V_newton is
                # then detached and loss.backward() would crash.
                # The same guard exists in the PINN branch above; mirror
                # it here so pure-MSE mode is equally robust.
                if train and not loss.requires_grad:
                    p0 = next(model.parameters())
                    loss = loss + 0.0 * p0.norm()
                    print("[warn] MSE loss detached (Armijo rejected all steps); applied zero-grad guard.")

            exact_phys = None
            if args.exact_physics_weight > 0.0 or (
                diagnose_gradients
                and len(gradient_supervised_losses) < args.report_gradient_alignment_batches
            ):
                exact_phys = power_flow_residual_loss(
                    Y_exact.to(torch.complex128),
                    Vpred,
                    Sstart_metric.to(torch.complex128),
                    bus_type,
                    sizes=n_nodes_per_graph,
                    residual_norm=args.exact_physics_residual_norm,
                    norm_eps=args.physics_norm_eps,
                )
                if args.exact_physics_weight > 0.0:
                    loss = loss + args.exact_physics_weight * exact_phys

            if (
                diagnose_gradients
                and len(gradient_supervised_losses) < args.report_gradient_alignment_batches
            ):
                _record_gradient_alignment(
                    gradient_alignment, supervised_mse, exact_phys
                )
                gradient_supervised_losses.append(float(supervised_mse.detach().item()))
                gradient_physics_losses.append(float(exact_phys.detach().item()))

            residual_metrics = compute_power_flow_residual_metrics(
                Y_exact,
                Vpred,
                Sstart_metric,
                bus_type,
                n_nodes_per_graph=n_nodes_per_graph,
                S_base=S_base_metric,
            )

            diagnostic_model = unwrap(model)
            if getattr(diagnostic_model, "global_context", None) is not None:
                global_gate_values.append(float(diagnostic_model.last_global_gate))
                global_entropy_values.append(
                    float(diagnostic_model.last_global_attention_entropy)
                )
                if diagnostic_model.global_context_gate_mode == "sdpa_sigmoid":
                    global_inner_gate_values.append(
                        float(diagnostic_model.last_global_inner_gate_mean)
                    )
                    global_inner_gate_low_values.append(
                        float(diagnostic_model.last_global_inner_gate_low_fraction)
                    )
            if getattr(diagnostic_model, "last_range_diagnostics", None) is not None:
                range_diagnostics_values.append(
                    diagnostic_model.last_range_diagnostics
                )
            for status, alpha in getattr(diagnostic_model, "last_armijo_trace", []):
                if status in armijo_status_counts:
                    armijo_status_counts[status] += 1
                armijo_alpha_values.append(float(alpha))

            if (
                record_step_diagnostics
                and range_step_batches_done < args.range_step_diagnostic_batches
            ):
                step_states = getattr(diagnostic_model, "last_step_states", [])
                if len(step_states) != K:
                    raise RuntimeError(
                        f"expected {K} recorded RANGE states, got {len(step_states)}"
                    )
                for step_index, step_state in enumerate(step_states):
                    dmag_step = step_state[..., 0] - Vnewton[..., 0]
                    dang_step = torch.atan2(
                        torch.sin(step_state[..., 1] - Vnewton[..., 1]),
                        torch.cos(step_state[..., 1] - Vnewton[..., 1]),
                    )
                    range_step_vm_sse[step_index] += float(
                        dmag_step.square().sum().item()
                    )
                    range_step_va_sse[step_index] += float(
                        dang_step.square().sum().item()
                    )
                    range_step_value_count[step_index] += int(dmag_step.numel())
                    step_residual = compute_power_flow_residual_metrics(
                        Y_exact,
                        step_state,
                        Sstart_metric,
                        bus_type,
                        n_nodes_per_graph=n_nodes_per_graph,
                    )
                    range_step_dp_values[step_index].append(
                        step_residual["dp_abs_valid"].cpu()
                    )
                    range_step_dq_values[step_index].append(
                        step_residual["dq_abs_valid"].cpu()
                    )
                range_step_batches_done += 1
                range_step_cases += B_eff

            if train:
                # Clipping and the optimiser step happen inside, once per
                # accumulation group; with ACCUM_STEPS == 1 this is the
                # original per-batch update.
                accum.backward(loss)

            sum_loss += loss.item() * B_eff
            sum_mse += mse.item() * B_eff
            sum_mse_mag += mse_mag.item() * B_eff
            sum_mse_ang += mse_ang.item() * B_eff
            sum_vm_sse += dmag.square().sum().item()
            sum_va_sse += dang.square().sum().item()
            sum_vm_target += Vnewton[..., 0].sum().item()
            sum_va_target += Vnewton[..., 1].sum().item()
            sum_vm_target_sq += Vnewton[..., 0].square().sum().item()
            sum_va_target_sq += Vnewton[..., 1].square().sum().item()
            r2_value_count += int(Vnewton[..., 0].numel())
            if per_bus_ok:
                V_bus = _per_bus_view(Vnewton, n_nodes_per_graph)
                if V_bus is None:
                    per_bus_ok = False
                else:
                    vm_b = V_bus[..., 0].double().sum(dim=0)
                    sin_b = torch.sin(V_bus[..., 1].double()).sum(dim=0)
                    cos_b = torch.cos(V_bus[..., 1].double()).sum(dim=0)
                    if per_bus_vm_sum is None:
                        per_bus_vm_sum = vm_b
                        per_bus_sin_sum = sin_b
                        per_bus_cos_sum = cos_b
                    elif per_bus_vm_sum.shape != vm_b.shape:
                        per_bus_ok = False
                    else:
                        per_bus_vm_sum += vm_b
                        per_bus_sin_sum += sin_b
                        per_bus_cos_sum += cos_b
                    per_bus_n_graphs += int(V_bus.shape[0])
            sum_max_dp_pu += residual_metrics["max_dp_pu"].sum().item()
            sum_max_dq_pu += residual_metrics["max_dq_pu"].sum().item()
            dp_dist_values.append(residual_metrics["dp_abs_valid"].detach().cpu())
            dq_dist_values.append(residual_metrics["dq_abs_valid"].detach().cpu())
            if "max_dp_mva" in residual_metrics:
                sum_max_dp_mva += residual_metrics["max_dp_mva"].sum().item()
                sum_max_dq_mva += residual_metrics["max_dq_mva"].sum().item()

            # Per-case convergence: pandapower-style ||F||_inf < tol applied
            # per case (both masked maxima below the engineering tolerance).
            _ctol = args.convergence_tol_pu
            _conv = (residual_metrics["max_dp_pu"] < _ctol) & (residual_metrics["max_dq_pu"] < _ctol)
            n_converged += int(_conv.sum().item())
            if kol_diag is not None:
                kol_initial_values.append(kol_diag["initial_max_mismatch"].detach().cpu())
                kol_final_values.append(kol_diag["final_max_mismatch"].detach().cpu())
                kol_converged_values.append(kol_diag["converged"].detach().cpu())
                kol_iteration_values.append(kol_diag["iterations"].detach().cpu())
            helm_module = getattr(model, "helm", None)
            helm_diag = getattr(helm_module, "last_diagnostics", None)
            if helm_diag:
                helm_residual_values.append(helm_diag["max_mismatch"].detach().cpu())
                helm_order_values.append(helm_diag["selected_order"].detach().cpu())
            if (
                diagnostic_only
                and len(gradient_supervised_losses)
                >= args.report_gradient_alignment_batches
            ):
                break

    if train:
        # A trailing partial group would otherwise be dropped.
        accum.flush()
    diagnostic_model.record_step_states = False
    if diagnose_gradients:
        _print_gradient_alignment(
            gradient_alignment,
            gradient_supervised_losses,
            gradient_physics_losses,
        )

    # Each rank holds only its shard's totals; sum them, and the counts they
    # divide by, before any mean is taken.
    (sum_loss, sum_mse, sum_mse_mag, sum_mse_ang, sum_max_dp_pu, sum_max_dq_pu,
     sum_max_dp_mva, sum_max_dq_mva, sum_vm_sse, sum_va_sse, sum_vm_target,
     sum_va_target, sum_vm_target_sq, sum_va_target_sq, _r2n, _nc, _ng) = reduce_sums(
        [sum_loss, sum_mse, sum_mse_mag, sum_mse_ang, sum_max_dp_pu, sum_max_dq_pu,
         sum_max_dp_mva, sum_max_dq_mva, sum_vm_sse, sum_va_sse, sum_vm_target,
         sum_va_target, sum_vm_target_sq, sum_va_target_sq, r2_value_count,
         n_converged, n_graphs_total], ddp)
    n_converged, n_graphs_total = int(round(_nc)), int(round(_ng))
    r2_value_count = int(round(_r2n))

    mean_loss = sum_loss / max(n_graphs_total, 1)
    convergence_rate = n_converged / max(n_graphs_total, 1)
    mean_mse = sum_mse / max(n_graphs_total, 1)
    mean_mse_mag = sum_mse_mag / max(n_graphs_total, 1)
    mean_mse_ang = sum_mse_ang / max(n_graphs_total, 1)
    mean_max_dp_pu = sum_max_dp_pu / max(n_graphs_total, 1)
    mean_max_dq_pu = sum_max_dq_pu / max(n_graphs_total, 1)
    mean_max_dp_mva = sum_max_dp_mva / max(n_graphs_total, 1)
    mean_max_dq_mva = sum_max_dq_mva / max(n_graphs_total, 1)
    residual_dist = finalize_residual_distribution(
        dp_dist_values,
        dq_dist_values,
        tol_pu=args.residual_tol_pu,
    )
    # Per-case convergence rate (pandapower-style ||F||_inf < tol, per case).
    # Carried inside residual_dist so no call-site signatures change.
    residual_dist["convergence_rate"] = convergence_rate
    residual_dist["convergence_tol_pu"] = args.convergence_tol_pu
    residual_dist["n_converged"] = n_converged
    residual_dist["n_cases"] = n_graphs_total
    # Two R^2 baselines; see the per_bus_* buffers above for why they differ.
    # "global" subtracts one mean over every bus-scenario entry, "per-bus"
    # subtracts each bus's own mean.  The per-bus one is reported as vm_r2 /
    # va_r2 because it is what the report tables use and the only one that
    # asks whether the model tracks the scenario; the global one is kept under
    # vm_r2_global / va_r2_global so older logs stay comparable.
    vm_sst_global = sum_vm_target_sq - sum_vm_target * sum_vm_target / max(r2_value_count, 1)
    va_sst_global = sum_va_target_sq - sum_va_target * sum_va_target / max(r2_value_count, 1)
    residual_dist["vm_r2_global"] = 1.0 - sum_vm_sse / max(vm_sst_global, 1e-30)
    residual_dist["va_r2_global"] = 1.0 - sum_va_sse / max(va_sst_global, 1e-30)

    per_bus_ok = bool(per_bus_ok) and per_bus_vm_sum is not None
    if ddp is not None and getattr(ddp, "distributed", False):
        import torch.distributed as dist

        ok_t = torch.tensor([1.0 if per_bus_ok else 0.0], dtype=torch.float64,
                            device=ddp.device)
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        per_bus_ok = bool(ok_t.item() > 0.5)
        if per_bus_ok:
            stacked = torch.stack([per_bus_vm_sum, per_bus_sin_sum, per_bus_cos_sum])
            dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
            per_bus_vm_sum, per_bus_sin_sum, per_bus_cos_sum = stacked.unbind(0)
            n_t = torch.tensor([float(per_bus_n_graphs)], dtype=torch.float64,
                               device=ddp.device)
            dist.all_reduce(n_t, op=dist.ReduceOp.SUM)
            per_bus_n_graphs = int(round(float(n_t.item())))

    if per_bus_ok and per_bus_n_graphs > 1:
        ns = float(per_bus_n_graphs)
        # Exact: sum_i sum_s y^2 - sum_i (sum_s y)^2 / n_s.
        vm_sst = float(sum_vm_target_sq
                       - (per_bus_vm_sum.square() / ns).sum().item())
        # Angles use each bus's circular mean, so a bus near +/-pi does not
        # acquire a spurious spread.  The wrapped deviations cannot be
        # recovered from running sums, so this uses the circular variance
        # 2(1 - Rbar).  Measured against the wrapped RMS it is low by 0.19% at
        # a 7 deg per-bus spread (GBnetwork) and by 1.5% at 20 deg (case300),
        # growing for wider spreads.  It errs low, which shrinks the
        # denominator and makes va_r2 slightly pessimistic rather than
        # flattering, so it is the safe direction to be wrong in.
        rbar = torch.hypot(per_bus_sin_sum / ns, per_bus_cos_sum / ns).clamp(max=1.0)
        va_sst = float((2.0 * (1.0 - rbar)).sum().item() * ns)
        residual_dist["vm_r2"] = 1.0 - sum_vm_sse / max(vm_sst, 1e-30)
        residual_dist["va_r2"] = 1.0 - sum_va_sse / max(va_sst, 1e-30)
        residual_dist["r2_baseline"] = "per_bus"
    else:
        # Mixed-size batches: "the same bus" is undefined, so fall back and say
        # so rather than silently reporting the looser number as if it were the
        # per-bus one.
        residual_dist["vm_r2"] = residual_dist["vm_r2_global"]
        residual_dist["va_r2"] = residual_dist["va_r2_global"]
        residual_dist["r2_baseline"] = "global"
    if global_gate_values:
        residual_dist["global_gate_mean"] = float(np.mean(global_gate_values))
        residual_dist["global_attention_entropy_mean"] = float(
            np.mean(global_entropy_values)
        )
        if global_inner_gate_values:
            residual_dist["global_inner_gate_mean"] = float(
                np.mean(global_inner_gate_values)
            )
            residual_dist["global_inner_gate_low_fraction"] = float(
                np.mean(global_inner_gate_low_values)
            )
    if range_diagnostics_values:
        range_scalar_names = (
            "aggregation_entropy",
            "broadcast_entropy",
            "master_attention_cosine",
            "master_embedding_similarity",
            "effective_masters",
            "self_loop_fraction",
        )
        for name in range_scalar_names:
            residual_dist[f"range_{name}_mean"] = float(np.mean([
                diagnostics[f"{name}_mean"]
                for diagnostics in range_diagnostics_values
            ]))
            residual_dist[f"range_step_{name}"] = np.mean(
                np.asarray([
                    diagnostics[f"step_{name}"]
                    for diagnostics in range_diagnostics_values
                ], dtype=np.float64),
                axis=0,
            ).tolist()
        residual_dist["range_token_utilization_mean"] = np.mean(
            np.asarray([
                diagnostics["token_utilization_mean"]
                for diagnostics in range_diagnostics_values
            ], dtype=np.float64),
            axis=0,
        ).tolist()
        residual_dist["range_step_token_utilization"] = np.mean(
            np.asarray([
                diagnostics["step_token_utilization"]
                for diagnostics in range_diagnostics_values
            ], dtype=np.float64),
            axis=0,
        ).tolist()
    if range_step_batches_done:
        step_diagnostics = []
        for step_index in range(K):
            dp_step = torch.cat(range_step_dp_values[step_index]).float()
            dq_step = torch.cat(range_step_dq_values[step_index]).float()
            count = max(int(range_step_value_count[step_index]), 1)
            step_diagnostics.append({
                "k": step_index + 1,
                "vm_rmse": math.sqrt(range_step_vm_sse[step_index] / count),
                "va_rmse_deg": math.sqrt(range_step_va_sse[step_index] / count)
                * (180.0 / math.pi),
                "median_dp": _safe_quantile(dp_step, 0.50),
                "p95_dp": _safe_quantile(dp_step, 0.95),
                "rmse_dp": torch.sqrt(dp_step.square().mean()).item(),
                "max_dp": dp_step.max().item(),
                "median_dq": _safe_quantile(dq_step, 0.50),
                "p95_dq": _safe_quantile(dq_step, 0.95),
                "rmse_dq": torch.sqrt(dq_step.square().mean()).item(),
                "max_dq": dq_step.max().item(),
            })
        residual_dist["range_step_diagnostics"] = step_diagnostics
        residual_dist["range_step_diagnostic_cases"] = range_step_cases
    armijo_total = sum(armijo_status_counts.values())
    if armijo_total:
        residual_dist["armijo_acceptance_rate"] = (
            armijo_status_counts["accept"] + armijo_status_counts["fallback"]
        ) / armijo_total
        residual_dist["armijo_forced_rate"] = (
            armijo_status_counts["safe"] / armijo_total
        )
        residual_dist["armijo_reject_rate"] = (
            armijo_status_counts["discard"] + armijo_status_counts["reject"]
        ) / armijo_total
        residual_dist["armijo_mean_alpha"] = float(np.mean(armijo_alpha_values))
    if kol_final_values:
        kol_initial = torch.cat(kol_initial_values).float()
        kol_final = torch.cat(kol_final_values).float()
        kol_converged = torch.cat(kol_converged_values).float()
        kol_iterations = torch.cat(kol_iteration_values).float()
        residual_dist["kol_initial_mean"] = kol_initial.mean().item()
        residual_dist["kol_final_mean"] = kol_final.mean().item()
        residual_dist["kol_certification_rate"] = kol_converged.mean().item()
        residual_dist["kol_mean_iterations"] = kol_iterations.mean().item()
        residual_dist["kol_tolerance"] = args.kol_tol
    if helm_residual_values:
        helm_residual = torch.cat(helm_residual_values).float()
        helm_orders = torch.cat(helm_order_values).float()
        residual_dist["helm_mean_max_mismatch"] = helm_residual.mean().item()
        residual_dist["helm_max_mismatch"] = helm_residual.max().item()
        residual_dist["helm_mean_selected_order"] = helm_orders.mean().item()
    return (
        mean_loss,
        mean_mse,
        mean_mse_mag,
        mean_mse_ang,
        mean_max_dp_pu,
        mean_max_dq_pu,
        mean_max_dp_mva,
        mean_max_dq_mva,
        residual_dist,
    )


# ------------------------------------------------------------------
# Training / validation
# ------------------------------------------------------------------
if "train" in args.mode:
    train_loss_hist, train_rmse_hist = [], []
    train_rmse_mag_hist, train_rmse_ang_hist_deg = [], []

    val_loss_hist, val_rmse_hist = [], []
    val_rmse_mag_hist, val_rmse_ang_hist_deg = [], []

    best_val_loss = float('inf')

    if args.skip_initial_eval:
        print("Initial metrics before training: skipped (--skip_initial_eval)")
    else:
        print("Initial metrics before training:")
        (
            train_loss,
            train_mse,
            train_mse_mag,
            train_mse_ang,
            train_max_dp_pu,
            train_max_dq_pu,
            _train_max_dp_mva,
            _train_max_dq_mva,
            _train_residual_dist,
        ) = run_epoch(train_loader, train=False, pinn=PINN)
        train_rmse = math.sqrt(train_mse)
        train_rmse_mag = math.sqrt(train_mse_mag)
        train_rmse_ang_deg = math.sqrt(train_mse_ang) * (180.0 / math.pi)

        (
            val_loss,
            val_mse,
            val_mse_mag,
            val_mse_ang,
            val_max_dp_pu,
            val_max_dq_pu,
            _val_max_dp_mva,
            _val_max_dq_mva,
            val_residual_dist,
        ) = run_epoch(val_loader, train=False, pinn=PINN)
        val_rmse = math.sqrt(val_mse)
        val_rmse_mag = math.sqrt(val_mse_mag)
        val_rmse_ang_deg = math.sqrt(val_mse_ang) * (180.0 / math.pi)

        print(
            f"Epoch   0 | "
            f"train loss {train_loss:.4e}  rmse {train_rmse:.4e} "
            f"(mag {train_rmse_mag:.4e}, ang {train_rmse_ang_deg:.4e}°) "
            f"{format_residual_summary(train_max_dp_pu, train_max_dq_pu)} | "
            f"valid loss {val_loss:.4e}  rmse {val_rmse:.4e} "
            f"(mag {val_rmse_mag:.4e}, ang {val_rmse_ang_deg:.4e}°) "
            f"{format_residual_summary(val_max_dp_pu, val_max_dq_pu)} "
            f"{format_residual_distribution_compact(val_residual_dist)}"
        )

    for epoch in range(1, EPOCHS + 1):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)   # else every epoch reuses one shuffle
        t0 = time.time()

        (
            train_loss,
            train_mse,
            train_mse_mag,
            train_mse_ang,
            train_max_dp_pu,
            train_max_dq_pu,
            _train_max_dp_mva,
            _train_max_dq_mva,
            _train_residual_dist,
        ) = run_epoch(train_loader, train=True, pinn=PINN)
        train_rmse = math.sqrt(train_mse)
        train_rmse_mag = math.sqrt(train_mse_mag)
        train_rmse_ang_deg = math.sqrt(train_mse_ang) * (180.0 / math.pi)

        train_loss_hist.append(train_loss)
        train_rmse_hist.append(train_rmse)
        train_rmse_mag_hist.append(train_rmse_mag)
        train_rmse_ang_hist_deg.append(train_rmse_ang_deg)

        if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
            (
                val_loss,
                val_mse,
                val_mse_mag,
                val_mse_ang,
                val_max_dp_pu,
                val_max_dq_pu,
                _val_max_dp_mva,
                _val_max_dq_mva,
                val_residual_dist,
            ) = run_epoch(val_loader, train=False, pinn=PINN)
            val_rmse = math.sqrt(val_mse)
            val_rmse_mag = math.sqrt(val_mse_mag)
            val_rmse_ang_deg = math.sqrt(val_mse_ang) * (180.0 / math.pi)

            val_loss_hist.append(val_loss)
            val_rmse_hist.append(val_rmse)
            val_rmse_mag_hist.append(val_rmse_mag)
            val_rmse_ang_hist_deg.append(val_rmse_ang_deg)

            print(
                f"Epoch {epoch:3d} | "
                f"train loss {train_loss:.4e}  rmse {train_rmse:.4e} "
                f"(mag {train_rmse_mag:.4e}, ang {train_rmse_ang_deg:.4e}°) "
                f"{format_residual_summary(train_max_dp_pu, train_max_dq_pu)} | "
                f"valid loss {val_loss:.4e}  rmse {val_rmse:.4e} "
                f"(mag {val_rmse_mag:.4e}, ang {val_rmse_ang_deg:.4e}°) "
                f"{format_residual_summary(val_max_dp_pu, val_max_dq_pu)} | "
                f"{format_residual_distribution_compact(val_residual_dist)} | "
                f"time {time.time() - t0:.2f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if ddp.is_main:
                    torch.save(unwrap(model).state_dict(), BEST_CKPT_PATH)
                    print(f"  ↳ checkpoint saved to {BEST_CKPT_PATH}")
        else:
            print(
                f"Epoch {epoch:3d} | "
                f"train loss {train_loss:.4e}  rmse {train_rmse:.4e} "
                f"(mag {train_rmse_mag:.4e}, ang {train_rmse_ang_deg:.4e}°) "
                f"{format_residual_summary(train_max_dp_pu, train_max_dq_pu)} | "
                f"time {time.time() - t0:.2f}s"
            )

    import matplotlib.pyplot as plt

    epochs = range(1, len(train_loss_hist) + 1)

    if PINN:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_loss_hist, label="Train Physics Loss")
        plt.plot(epochs[:len(val_loss_hist)], val_loss_hist, label="Validation Physics Loss")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("PINN: Physics Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./results/plots/{RUNNAME}_physics_loss.png")
        plt.clf()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_rmse_hist, label="Train RMSE")
    plt.plot(epochs[:len(val_rmse_hist)], val_rmse_hist, label="Val RMSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Supervised RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./results/plots/{RUNNAME}_rmse_total.png")
    plt.clf()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, train_rmse_mag_hist, label="Train |V|")
    ax[0].plot(epochs[:len(val_rmse_mag_hist)], val_rmse_mag_hist, label="Val |V|")
    ax[0].set_title("Magnitude RMSE")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(epochs, train_rmse_ang_hist_deg, label="Train θ (deg)")
    ax[1].plot(epochs[:len(val_rmse_ang_hist_deg)], val_rmse_ang_hist_deg, label="Val θ (deg)")
    ax[1].set_title("Angle RMSE (degrees)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    fig.suptitle("Magnitude vs Angle RMSE")
    fig.tight_layout()
    fig.savefig(f"./results/plots/{RUNNAME}_rmse_components.png")
    plt.close(fig)


# ------------------------------------------------------------------
# NR-polish: warm-start a Newton-Raphson solver from the model prediction
# and count iterations to convergence. Two engines selectable.
# ------------------------------------------------------------------
def _busmap_own(bus_type_np):
    """Remap to the own-NR convention (1=slack, 2=PV, 3=PQ) regardless of how
    PQ is coded in the source (0 or 3)."""
    bt = np.full(bus_type_np.shape, 3, dtype=np.int64)
    bt[bus_type_np == 1] = 1
    bt[bus_type_np == 2] = 2
    return bt


def _nr_polish_own(Ybus, Sbus, V0, bus_type_np, tol, max_iter):
    """Use the data-generation NR (newton_raphson_improved.newtonrapson)."""
    import sys as _sys
    if args.nr_impl_path not in _sys.path:
        _sys.path.insert(0, args.nr_impl_path)
    from newton_raphson_improved import newtonrapson
    bt = _busmap_own(bus_type_np)
    _, _, _, diag = newtonrapson(
        bt, Ybus.astype(np.complex128), Sbus.astype(np.complex128),
        V0.astype(np.complex128),
        K=int(max_iter), diagnose=True, return_diagnostics=True,
        convergence_mode="misinf", mismatch_tol=float(tol),
    )
    return int(diag.get("iterations", 0)), bool(diag.get("converged", False))


def _nr_polish_pypower(Ybus, Sbus, V0, bus_type_np, tol, max_iter):
    """Standalone polar Newton-Raphson identical to pypower/pandapower's
    newtonpf core, with pandapower's convergence test ||F||_inf < tol."""
    Ybus = Ybus.astype(np.complex128)
    Sbus = Sbus.astype(np.complex128)
    V = V0.astype(np.complex128).copy()
    Va = np.angle(V); Vm = np.abs(V)
    ref = np.where(bus_type_np == 1)[0]
    pv = np.where(bus_type_np == 2)[0]
    pq = np.where((bus_type_np != 1) & (bus_type_np != 2))[0]
    pvpq = np.r_[pv, pq]
    npvpq, npq = len(pvpq), len(pq)

    def mismatch(V):
        Scalc = V * np.conj(Ybus @ V)
        dS = Scalc - Sbus
        return np.r_[dS[pvpq].real, dS[pq].imag]

    F = mismatch(V)
    converged = np.linalg.norm(F, np.inf) < tol
    i = 0
    while (not converged) and i < int(max_iter):
        i += 1
        Ibus = Ybus @ V
        diagV = np.diag(V)
        diagIbus = np.diag(Ibus)
        diagVnorm = np.diag(V / np.abs(V))
        dS_dVm = diagV @ np.conj(Ybus @ diagVnorm) + np.conj(diagIbus) @ diagVnorm
        dS_dVa = 1j * diagV @ np.conj(diagIbus - Ybus @ diagV)
        J11 = dS_dVa[np.ix_(pvpq, pvpq)].real
        J12 = dS_dVm[np.ix_(pvpq, pq)].real
        J21 = dS_dVa[np.ix_(pq, pvpq)].imag
        J22 = dS_dVm[np.ix_(pq, pq)].imag
        J = np.block([[J11, J12], [J21, J22]])
        try:
            dx = -np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            return i, False
        Va[pvpq] += dx[:npvpq]
        Vm[pq] += dx[npvpq:npvpq + npq]
        V = Vm * np.exp(1j * Va)
        Va, Vm = np.angle(V), np.abs(V)
        F = mismatch(V)
        converged = np.linalg.norm(F, np.inf) < tol
    return i, bool(converged)


def run_nr_polish_eval():
    """Warm-start NR from model prediction vs flat start; report iterations."""
    polish = _nr_polish_own if args.nr_polish_solver == "own" else _nr_polish_pypower
    print(f"\n[nr-polish] solver={args.nr_polish_solver} tol={args.nr_polish_tol:.1e} "
          f"max_iter={args.nr_polish_max_iter} cap={args.nr_polish_max_cases} cases")

    # batch_size=1 single-graph loader so each case is one grid (no blockdiag split)
    polish_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                               collate_fn=collate_blockdiag)
    iters_pred, iters_flat = [], []
    conv_pred, conv_flat = 0, 0
    n_done = 0
    model.eval()
    with torch.no_grad():
        for batch in polish_loader:
            if n_done >= args.nr_polish_max_cases:
                break
            bus_type = batch["bus_type"].to(device)
            Branch_f_bus = batch["Branch_f_bus"].to(device)
            Branch_t_bus = batch["Branch_t_bus"].to(device)
            Branch_status = batch["Branch_status"].to(device)
            Branch_tau = batch["Branch_tau"].to(device)
            Branch_shift_deg = batch["Branch_shift_deg"].to(device)
            Branch_y_series_from = batch["Branch_y_series_from"].to(device)
            Branch_y_series_to = batch["Branch_y_series_to"].to(device)
            Branch_y_series_ft = batch["Branch_y_series_ft"].to(device)
            Branch_y_shunt_from = batch["Branch_y_shunt_from"].to(device)
            Branch_y_shunt_to = batch["Branch_y_shunt_to"].to(device)
            Is_trafo = batch["Is_trafo"].to(device)
            Y_shunt_bus = batch["Y_shunt_bus"].to(device)
            if BLOCK_DIAG and "sizes" in batch:
                n_nodes_per_graph = batch["sizes"].to(device)
            else:
                n_nodes_per_graph = None
            Sstart = batch["S_start"].to(device)
            Vstart = batch["V_start"].to(device)
            vn_log = batch["vn_log"].to(device) if "vn_log" in batch else None
            Y = batch.get("Ybus", None)
            if Y is not None:
                Y = Y.to(device)

            Sstart_exact = Sstart
            Vstart_exact = Vstart

            Yd = ensure_dense_y_for_metrics(
                Y, bus_type, Branch_f_bus, Branch_t_bus, Branch_status,
                Branch_tau, Branch_shift_deg, Branch_y_series_from,
                Branch_y_series_to, Branch_y_series_ft, Branch_y_shunt_from,
                Branch_y_shunt_to, Y_shunt_bus,
            )
            if args.dataset_complex_dtype == "complex128":
                Branch_tau = Branch_tau.float()
                Branch_shift_deg = Branch_shift_deg.float()
                Branch_y_series_from = Branch_y_series_from.to(torch.complex64)
                Branch_y_series_to = Branch_y_series_to.to(torch.complex64)
                Branch_y_series_ft = Branch_y_series_ft.to(torch.complex64)
                Branch_y_shunt_from = Branch_y_shunt_from.to(torch.complex64)
                Branch_y_shunt_to = Branch_y_shunt_to.to(torch.complex64)
                Y_shunt_bus = Y_shunt_bus.to(torch.complex64)
                if Y is not None:
                    Y = Y.to(torch.complex64)
                Sstart = Sstart.to(torch.complex64)
                Vstart = Vstart.float()
                if vn_log is not None:
                    vn_log = vn_log.float()
            out = model(
                bus_type, Branch_f_bus, Branch_t_bus, Branch_status,
                Branch_tau, Branch_shift_deg, Branch_y_series_from,
                Branch_y_series_to, Branch_y_series_ft, Branch_y_shunt_from,
                Branch_y_shunt_to, Is_trafo, Y, Sstart, Vstart,
                n_nodes_per_graph=n_nodes_per_graph, Y_shunt_bus=Y_shunt_bus,
                vn_log=vn_log,
            )
            Vpred = out[0] if isinstance(out, tuple) else out
            if known_operator is not None:
                Vpred, _ = known_operator(
                    Vpred,
                    Yd,
                    Sstart_exact,
                    bus_type,
                    V_fixed=Vstart_exact,
                    sizes=n_nodes_per_graph,
                    differentiable=False,
                )

            Yb_tensor = Yd[0] if Yd.dim() == 3 else Yd
            if Yb_tensor.is_sparse:
                Yb_tensor = Yb_tensor.to_dense()
            Yb = Yb_tensor.cpu().numpy()
            S_np = Sstart_exact[0].cpu().numpy()
            bt_np = bus_type[0].cpu().numpy()
            vp = Vpred[0].cpu().numpy()
            V0_pred = vp[:, 0] * np.exp(1j * vp[:, 1])
            vs = Vstart_exact[0].cpu().numpy()
            V0_flat = vs[:, 0] * np.exp(1j * vs[:, 1])

            ip, cp = polish(Yb, S_np, V0_pred, bt_np, args.nr_polish_tol, args.nr_polish_max_iter)
            iff, cf = polish(Yb, S_np, V0_flat, bt_np, args.nr_polish_tol, args.nr_polish_max_iter)
            iters_pred.append(ip); conv_pred += int(cp)
            iters_flat.append(iff); conv_flat += int(cf)
            n_done += 1

    if n_done == 0:
        print("[nr-polish] no cases evaluated."); return
    ip = np.array(iters_pred); iff = np.array(iters_flat)
    print(f"[nr-polish] cases={n_done}")
    print(f"[nr-polish] model warm-start : iters mean {ip.mean():.2f}  median {np.median(ip):.0f}  "
          f"max {ip.max()}  converged {conv_pred}/{n_done} ({conv_pred/n_done:.1%})")
    print(f"[nr-polish] input/DC start   : iters mean {iff.mean():.2f}  median {np.median(iff):.0f}  "
          f"max {iff.max()}  converged {conv_flat}/{n_done} ({conv_flat/n_done:.1%})")
    saved = iff.mean() - ip.mean()
    print(f"[nr-polish] iterations saved by surrogate (input/DC - model): {saved:.2f} "
          f"({saved/max(iff.mean(),1e-9):.1%} reduction)")


# ------------------------------------------------------------------
# Final test
# ------------------------------------------------------------------
if "valid" in args.mode and "train" not in args.mode:
    (
        valid_loss,
        valid_mse,
        valid_mse_mag,
        valid_mse_ang,
        valid_max_dp_pu,
        valid_max_dq_pu,
        valid_max_dp_mva,
        valid_max_dq_mva,
        valid_residual_dist,
    ) = run_epoch(val_loader, train=False, pinn=PINN)
    valid_rmse = math.sqrt(valid_mse)
    valid_rmse_mag = math.sqrt(valid_mse_mag)
    valid_rmse_ang_deg = math.sqrt(valid_mse_ang) * (180.0 / math.pi)
    print(
        f"\nValidation physics-loss : {valid_loss:.4e}"
        f" | total RMSE : {valid_rmse:.4e}"
        f" | |V| RMSE : {valid_rmse_mag:.4e}"
        f" | θ RMSE : {valid_rmse_ang_deg:.4e}°"
        f" | ΔP∞ : {valid_max_dp_pu:.4e} pu ({valid_max_dp_mva:.4e} MW)"
        f" | ΔQ∞ : {valid_max_dq_pu:.4e} pu ({valid_max_dq_mva:.4e} MVAr)"
    )
    print(format_residual_distribution_full(valid_residual_dist, tol_pu=args.residual_tol_pu))

if "test" in args.mode:
    if args.helm_uniform_path:
        print("[test] uniform HELM path: PIGNN bypassed; no checkpoint loaded.")
    elif os.path.exists(BEST_CKPT_PATH):
        barrier(ddp)
        unwrap(model).load_state_dict(torch.load(BEST_CKPT_PATH, map_location=device))
        print(f"[test] loaded best checkpoint: {BEST_CKPT_PATH}")
    else:
        print(f"[test] best checkpoint not found at {BEST_CKPT_PATH}; using current model weights.")

    if (
        args.report_gradient_alignment_batches > 0
        and args.gradient_alignment_split == "train"
    ):
        if not BLOCK_DIAG:
            raise ValueError(
                "fixed train-split gradient diagnostic currently requires --BLOCK_DIAG"
            )
        gradient_loader = DataLoader(
            train_ds,
            batch_size=BATCH,
            shuffle=False,
            collate_fn=collate_blockdiag,
        )
        print(
            f"[grad-align] split=train fixed_order=true "
            f"requested_batches={args.report_gradient_alignment_batches}"
        )
        run_epoch(
            gradient_loader,
            train=False,
            pinn=PINN,
            diagnose_gradients=True,
            diagnostic_only=True,
        )

    (
        test_loss,
        test_mse,
        test_mse_mag,
        test_mse_ang,
        test_max_dp_pu,
        test_max_dq_pu,
        test_max_dp_mva,
        test_max_dq_mva,
        test_residual_dist,
    ) = run_epoch(
        test_loader,
        train=False,
        pinn=PINN,
        diagnose_gradients=(
            args.report_gradient_alignment_batches > 0
            and args.gradient_alignment_split == "test"
        ),
        record_step_diagnostics=(args.global_context_mode == "range_post"),
    )

    test_rmse = math.sqrt(test_mse)
    test_rmse_mag = math.sqrt(test_mse_mag)
    test_rmse_ang_deg = math.sqrt(test_mse_ang) * (180.0 / math.pi)

    if not PINN:
        print(
            f"\nFinal test-set RMSE : {test_rmse:.4e}"
            f" (|V|: {test_rmse_mag:.4e}, θ: {test_rmse_ang_deg:.4e}°)"
        )
    print(
        f"\nTest exact-complex residuals"
        f" | total RMSE : {test_rmse:.4e}"
        f" | |V| RMSE : {test_rmse_mag:.4e}"
        f" | θ RMSE : {test_rmse_ang_deg:.4e}°"
        f" | mean case-max ΔP : {test_max_dp_pu:.4e} pu ({test_max_dp_mva:.4e} MW)"
        f" | mean case-max ΔQ : {test_max_dq_pu:.4e} pu ({test_max_dq_mva:.4e} MVAr)"
    )
    print(format_residual_distribution_full(test_residual_dist, tol_pu=args.residual_tol_pu))

    if args.report_nr_polish:
        run_nr_polish_eval()
