"""PIGNN-style train-only normalization for supervised PF voltage loss."""

import torch


def add_correction_target_norm_args(parser):
    parser.add_argument(
        "--correction_target_norm",
        choices=("none", "bus", "voltage_level"),
        default="none",
        help=(
            "Normalize supervised delta-|V| and wrapped delta-angle by their "
            "train-split standard deviations, matching PIGNN."
        ),
    )
    parser.add_argument(
        "--correction_target_norm_eps",
        type=float,
        default=1e-4,
        help="Minimum train-set standard deviation used for target normalization.",
    )


def estimate_correction_target_scales(loader, mode, eps):
    """Estimate PIGNN's per-bus/channel correction scales from training only."""
    if mode == "none":
        return None
    if eps <= 0.0:
        raise ValueError("--correction_target_norm_eps must be positive")

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
            if "sizes" in batch:
                sizes = [int(x) for x in batch["sizes"].tolist()]
                graphs = list(torch.split(delta.squeeze(0), sizes, dim=0))
                vn_graphs = (
                    list(torch.split(batch["vn_log"].cpu().double().squeeze(0), sizes, dim=0))
                    if "vn_log" in batch else [None] * len(graphs)
                )
            else:
                graphs = list(delta)
                vn_graphs = (
                    list(batch["vn_log"].cpu().double())
                    if "vn_log" in batch else [None] * len(graphs)
                )

            for graph, vn in zip(graphs, vn_graphs):
                n_bus = graph.shape[0]
                if sums is None:
                    sums = torch.zeros(n_bus, 2, dtype=torch.float64)
                    sums_sq = torch.zeros_like(sums)
                    counts = torch.zeros(n_bus, 1, dtype=torch.float64)
                    reference_vn = vn.clone() if vn is not None else None
                if n_bus != sums.shape[0]:
                    raise ValueError(
                        "--correction_target_norm requires one fixed grid/bus ordering"
                    )
                if mode == "voltage_level":
                    if vn is None:
                        raise ValueError("voltage_level normalization requires vn_log")
                    if not torch.allclose(vn, reference_vn, atol=1e-6, rtol=0.0):
                        raise ValueError("vn_log/bus ordering changes across scenarios")
                sums += graph
                sums_sq += graph.square()
                counts += 1.0

    variance = (sums_sq / counts - (sums / counts).square()).clamp_min(0.0)
    if mode == "bus":
        scales = variance.sqrt()
    else:
        levels = torch.round(reference_vn * 1e6) / 1e6
        scales = torch.empty_like(variance)
        for level in torch.unique(levels):
            mask = levels == level
            group_count = counts[mask].sum()
            group_mean = sums[mask].sum(dim=0) / group_count
            group_var = (
                sums_sq[mask].sum(dim=0) / group_count - group_mean.square()
            ).clamp_min(0.0)
            scales[mask] = group_var.sqrt()

    scales = scales.clamp_min(float(eps)).float()
    print(
        f"[target-whiten] mode={mode}, buses={scales.shape[0]}, "
        f"sigma_mag min/median/max={scales[:, 0].min():.3e}/"
        f"{scales[:, 0].median():.3e}/{scales[:, 0].max():.3e}, "
        f"sigma_ang min/median/max={scales[:, 1].min():.3e}/"
        f"{scales[:, 1].median():.3e}/{scales[:, 1].max():.3e}"
    )
    return scales


def correction_scales_for_batch(scales, vpred, n_nodes_per_graph):
    if scales is None:
        return None
    scale = scales.to(device=vpred.device, dtype=vpred.dtype)
    if n_nodes_per_graph is None:
        if vpred.shape[-2] != scale.shape[0]:
            raise ValueError("correction target scale does not match batch bus count")
        return scale.unsqueeze(0)
    sizes = [int(x) for x in n_nodes_per_graph.tolist()]
    if any(size != scale.shape[0] for size in sizes):
        raise ValueError("correction target scale does not match block-diagonal graph size")
    return scale.repeat(len(sizes), 1).unsqueeze(0)


def normalized_voltage_mse(dmag, dang, scales, unnormalized_angle_weight=1.0):
    """Return PIGNN's two equally weighted, dimensionless channel losses."""
    if scales is None:
        return torch.mean(dmag.square()) + float(unnormalized_angle_weight) * torch.mean(
            dang.square()
        )
    return torch.mean((dmag / scales[..., 0]).square()) + torch.mean(
        (dang / scales[..., 1]).square()
    )
