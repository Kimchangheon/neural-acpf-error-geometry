# Large artifacts: where they live and how to fetch them

Nothing in this section is stored in the repository. Corpora are tens of
gigabytes each and checkpoints are per-experiment, so both stay on the clusters
and are listed here by absolute path.

Two accounts are involved and they are **not** the same filesystem:

| Alias | Host | Account | Home |
|---|---|---|---|
| `helma`, `alex2` | `helma.nhr.fau.de`, `alex.nhr.fau.de` | `b313dc11` | `/home/hpc/b313dc/b313dc11` |
| `alex` | `alex.nhr.fau.de` | `iwi5295h` | `/home/hpc/iwi5/iwi5295h` |

`helma` and `alex2` share one home directory; `alex` is a separate account with
its own copy of the tree. Both reach the login nodes through the
`csnhr.nhr.fau.de` jump host, which intermittently refuses connections — retry
rather than treating a refusal as an outage.

Shorthands used below:

```
HELMA_REPO=/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
HELMA_VAULT=/home/vault/b313dc/b313dc11
ALEX_REPO=/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC
```

---

## 1. Scenario corpora (ppNR v2, 31 grids)

```
$HELMA_VAULT/data/pf/
```

Every file is one grid. The generator settings are encoded in the name:
`ppNR` means the label is pandapower's converged Newton–Raphson solution,
`ls0.60-1.40` the load-scaling range, `siNR` that the solver ran in SI units,
and `directSI` the storage convention. `_rg20` marks the 20-row-group layout the
lazy loader reads.

| Grid | Scenarios | Size | File |
|---|---:|---:|---|
| `case118` | 36,000 | 0.56 GiB | `case118_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case1354pegase` | 32,627 | 2.73 GiB | `case1354pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_32627_NR_branchrows_directSI_rg20.parquet` |
| `case14` | 36,000 | 0.09 GiB | `case14_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case145` | 36,000 | 0.13 GiB | `case145_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case1888rte` | 36,000 | 4.14 GiB | `case1888rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case24_ieee_rts` | 36,000 | 0.14 GiB | `case24_ieee_rts_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case2848rte` | 36,000 | 6.22 GiB | `case2848rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case2869pegase` | 35,998 | 6.32 GiB | `case2869pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_35998_NR_branchrows_directSI_rg20.parquet` |
| `case30` | 36,000 | 0.15 GiB | `case30_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case300` | 36,000 | 0.65 GiB | `case300_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case3120sp` | 35,465 | 6.84 GiB | `case3120sp_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_35465_NR_branchrows_directSI_rg20.parquet` |
| `case33bw` | 36,000 | 0.16 GiB | `case33bw_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case39` | 36,000 | 0.18 GiB | `case39_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case4gs` | 36,000 | 0.05 GiB | `case4gs_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case5` | 36,000 | 0.05 GiB | `case5_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case57` | 36,000 | 0.27 GiB | `case57_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case6470rte` | 25,551 | 9.84 GiB | `case6470rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_25551_NR_branchrows_directSI_rg20.parquet` |
| `case6495rte` | 23,332 | 9.01 GiB | `case6495rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_23332_NR_branchrows_directSI_rg20.parquet` |
| `case6515rte` | 20,135 | 7.80 GiB | `case6515rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_20135_NR_branchrows_directSI_rg20.parquet` |
| `case6ww` | 36,000 | 0.06 GiB | `case6ww_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case89pegase` | 36,000 | 0.48 GiB | `case89pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case9` | 36,000 | 0.06 GiB | `case9_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case9241pegase` | 18,280 | 10.39 GiB | `case9241pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_18280_NR_branchrows_directSI_rg20.parquet` |
| `case_ieee30` | 36,000 | 0.15 GiB | `case_ieee30_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `case_illinois200` | 36,000 | 0.83 GiB | `case_illinois200_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `ENTSO_E_RealGridTest` | 36,000 | 11.95 GiB | `ENTSO_E_RealGridTest_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_siNR_36000_NR_branchrows_directSI.parquet` |
| `GBnetwork` | 34,424 | 4.32 GiB | `GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_34424_NR_branchrows_directSI_rg20.parquet` |
| `GBreducednetwork` | 36,000 | 0.21 GiB | `GBreducednetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `iceland` | 36,000 | 0.71 GiB | `iceland_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_siNR_36000_NR_branchrows_directSI_rg20.parquet` |
| `LVN_heo1` | 36,000 | 1.36 GiB | `LVN_heo1_ppcY_backbone_dc_compile_cNR_ls0.60-1.40_u0clean_siNR_36000_NR_branchrows_directSI.parquet` |
| `SimBench` | 36,000 | 0.38 GiB | `SimBench_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_siNR_36000_NR_branchrows_directSI.parquet` |

**Total: 31 grids, 86.2 GiB.**

The vault also holds earlier and variant corpora under `data/pf/` subdirectories
(`largegrid_conditioning_20260824/`) and under
`$HELMA_VAULT/PIGNN-Attn-LS/data/{pf,pignn_hetero,nrtraj}/`. Those are **not**
what the paper uses; the 31 files listed above are.

### GBnetwork working copies

GBnetwork is the grid nearly every table is computed on, so staged copies exist
close to compute:

| Path | Size | Used by |
|---|---|---|
| `/home/hpc/b313dc/b313dc11/pb_data/GBnetwork.parquet` | 4.3 GiB | most GBnetwork scoring jobs |
| `/home/hpc/b313dc/b313dc11/data_staging/controlled_geometry_20260905/GBnetwork.parquet` | 4.3 GiB | controlled error-geometry campaign |
| `$HELMA_VAULT/PIGNN-Attn-LS/data/pignn_hetero/GBnetwork_ppcY_..._siNR_37022_....parquet` | 4.3 GiB | archival copy of the same corpus |

These three are byte-identical (verified by size and md5 during the campaign);
the duplication is staging, not different data.

### Distribution- and topology-shift corpora (Table `tab:shift`)

| Path | Size | Role |
|---|---|---|
| `/home/hpc/b313dc/b313dc11/data_staging/mix_n1_20260905/armA_control_only.parquet` | 0.83 GiB | control arm, base topology only |
| `/home/hpc/b313dc/b313dc11/data_staging/mix_n1_20260905/armB_control_plus_n1.parquet` | 0.85 GiB | control + N-1 outage arm |

The correlated-tilt (`gbcorr`) arm was run on the **`alex`** account. Its scoring
script reads `$R/data/control.parquet` and `$R/data/gbcorr.parquet`, where `$R`
is the job's own staging root — see
`reproduction/experiments/historical_local/gbcorr_csp_posthoc_alex2_20260910.sh` for the
exact staging block. The results are mirrored in
`reproduction/results/alex/gbcorr_csp_posthoc_20260910/`.

---

## 2. Checkpoints behind the main table

Table `tab:intervention` reports three model seeds per model. These are the
exact files each seed was scored from, recovered from the scoring scripts rather
than from memory.

### PIGNN-GC — 832 KiB each

```
$HELMA_REPO/results/g3_ref_s41_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s41_40_best_model.ckpt
$HELMA_REPO/results/g3_ref_s43_ablation_20260829/ckpt/pignn_global_GBnetwork_g3_ref_s43_40_best_model.ckpt
$HELMA_REPO/overlays/residual_split_20260829/pignn_global_GBnetwork_g3_s42_40_best_model.ckpt
```

The seed-42 row is the older frozen `g3` baseline from a different campaign, not
a matched `g3_ref` replicate. Some scoring scripts point at a
`overlays/residual_diagnostics_20260828/` copy of the same file, which no longer
exists; the `residual_split_20260829` copy is the surviving one.

### GridSFM — 117 MiB each

```
$HELMA_REPO/results/model_seed_replicates_20260907/ckpt/gridsfm/gridsfm_GBnetwork_s41_40_best.pt
$HELMA_REPO/results/model_seed_replicates_20260907/ckpt/gridsfm/gridsfm_GBnetwork_s43_40_best.pt
$HELMA_REPO/results/ckpt/pfv2_gridsfm_D_v2/pfv2_gridsfm_GBnetwork_b26_best.pt
```

Same caveat: the third file is the original frozen baseline
(`pfv2_gridsfm_D_v2`, SHA-256 beginning `0c40ed…d602ed9`) grouped with two
matched replicates, not a third matched replicate.

### GridFM-GraphKit — 154 MiB each

```
$HELMA_REPO/results/ckpt/gk_e120_20260910/k06_s41_e120_best.pt
$HELMA_REPO/results/ckpt/gk_e120_20260910/k07_s43_e120_best.pt
$HELMA_REPO/results/ckpt/gk_e120_20260910/k08_s42_e120_best.pt
```

All three are 120-epoch runs trained under one protocol, so this is the one
model whose three rows are protocol-matched throughout.

### LUMINA — 9.6 MiB each

```
$HELMA_REPO/results/lumina_gbv06z_seedrep_helma_h100_20260914/ckpt/s41/gbv06z_control_zerohead_h100_s41_best.pt
$HELMA_REPO/results/lumina_gbv06z_seedrep_helma_h100_20260914/ckpt/s43/gbv06z_control_zerohead_h100_s43_best.pt
$HELMA_REPO/results/ckpt/gb_vsweep2_20260911/gbv06z_control_zerohead_best.pt
```

The seed-42 checkpoint is the original `gb_vsweep2` run; the other two are the
H100 seed replicates. An A40 replicate set also exists at
`$HELMA_REPO/results/lumina_gbv06z_seedrep_20260913/ckpt/s{41,43}/` and was used
for the earlier A40 scoring pass.

---

## 3. Fetching

Single file:

```bash
scp helma:/home/hpc/b313dc/b313dc11/pb_data/GBnetwork.parquet .
```

A whole result group, excluding the heavy binaries:

```bash
rsync -a --prune-empty-dirs \
  --include='*/' --include='*.json' --include='*.csv' --include='*.md' --exclude='*' \
  helma:/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/results/<group>/ ./results/helma/<group>/
```

That is exactly what `tools/assemble_release.py` does, with a 5 MB per-file cap,
so re-running it refreshes the small artifacts without pulling corpora or
checkpoints. It never overwrites a differing file; it raises instead.

The corpora are large enough that a full mirror is 86 GiB. Fetch per grid.
