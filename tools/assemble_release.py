#!/usr/bin/env python3
"""Maintainer-only, non-destructive export of the paper's research snapshot.

Preserves imported sources byte-for-byte. Large artifacts are never fetched.
Existing differing files are left intact; remote/local versions live separately.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

DEST = Path(__file__).resolve().parents[1]
# The research snapshot lives under reproduction/; csp/ is hand-written
# and is never touched by this script.
ARCHIVE = DEST / 'reproduction'
SRC = Path('/Users/changhunkim/PycharmProjects/PIGNN-Attn-LS')
PPC = SRC / 'PIGNN-Attn-LS-PPC'
REMOTE = '/home/hpc/b313dc/b313dc11/PIGNN-Attn-LS'
ALEX = '/home/hpc/iwi5/iwi5295h/PIGNN-Attn-LS'
ENTRIES = '''controlled_error_geometry output_intervention_metrics csp_projection_block_ablation csp_posthoc_diagnostics jacobian_subspace_alignment jacobian_subspace_reproducibility off_subspace_conditional_association basis_reconstruction_matched_controls compare_solution_basis_controls evaluate_scalar_shrinkage gbcorr_csp_posthoc nr1_multiprocess_baseline nr1_metric_complete_rescore select_nr1_eta select_nr1_eta_graphkit benchmark_inference_walltime summarize_gridsfm_31grid_csp16 train_valid_test train_valid_test_gridfm train_valid_test_gridsfm train_valid_test_lumina rescore_slope_r2 summarize_gk_both_3seed'''.split()
GROUPS = '''controlled_geometry_20260905 model_seed_replicates_20260907 gbnetwork_known_state_csp_k16_20260909 graphkit_e120_fullstate_csp_k16_20260911 graphkit_e120_fullstate_csp_k16_alex_20260911 lumina_gbv06z_3seed_fullstate_csp_k16_20260914 lumina_gbv06z_h100_3seed_fullstate_csp_k16_20260914 gbnetwork_inference_walltime_h100_20260912 lumina_gbv06z_h100_3seed_inference_20260914 gridsfm_31grid_csp16_20260907 basis_reconstruction_matched_20260914 off_subspace_conditional_association_20260914 csp_posthoc_diagnostics_gbnetwork_20260910 csp_posthoc_diagnostics_graphkit_e120_20260912 csp_fullstate_vs_restore_gbnetwork_20260910 csp_projection_block_ablation_norestore_20260910 csp_projection_block_ablation_graphkit_e120_norestore_20260912 jacobian_reproducibility_gbnetwork_20260910 jacobian_graphkit_e120_20260912 fixed_norm_directional_graphkit_e120_20260912 nr1_multiproc_gbnetwork_20260910 nr1_graphkit_e120_gbnetwork_a40_20260912 nr1_graphkit_e120_gbnetwork_20260912 nr1_lumina_gbv06z_3seed_20260914 pignn_g3_basis_controls_20260909_alex2 pignn_g3_shrinkage_20260909_alex2 mix_n1_20260905'''.split()
ALEX_GROUPS = '''csp_blocks_lumina_gbv06z_20260913 csp_posthoc_lumina_gbv06z_20260913 fixed_norm_lumina_gbv06z_20260913 jacobian_lumina_gbv06z_20260913 nr1_lumina_gbv06z_20260913 gbcorr_csp_posthoc_20260910'''.split()
EXT = {'.py','.sh','.sbatch','.json','.csv','.tsv','.md','.txt','.yaml','.yml','.toml','.cfg','.ini','.out','.err','.log','.tex','.bib','.bst','.sty','.pdf','.png','.svg'}
records = []

def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.read_bytes() != src.read_bytes():
        raise RuntimeError(f'Refusing to overwrite differing file: {dst}')
    shutil.copy2(src, dst)
    records.append({'source':str(src),'destination':str(dst.relative_to(DEST)), 'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})

def tree(src, dst):
    if not src.exists(): return
    for p in sorted(src.rglob('*')):
        if p.is_file() and not any(x in p.parts for x in ('.git','__pycache__','node_modules','build')) and p.suffix in EXT and p.stat().st_size <= 5_000_000:
            copy(p,dst/p.relative_to(src))

def remote(host, path, dst, suffixes=EXT):
    dst.mkdir(parents=True,exist_ok=True)
    args=['rsync','-a','-e','ssh -o ProxyJump=none -o BatchMode=yes -o ConnectTimeout=15','--ignore-existing','--prune-empty-dirs','--max-size=5m','--exclude=.git/','--exclude=__pycache__/','--include=*/']
    args += [f'--include=*{s}' for s in sorted(suffixes)]
    args += ['--include=LICENSE*','--include=NOTICE*','--exclude=*',f'{host}:{path}/',str(dst)+'/']
    r=subprocess.run(args, capture_output=True,text=True)
    print(f'{host}:{path}: {r.returncode}',flush=True)
    if r.returncode: print(r.stderr[-600:],flush=True)
    return {'host':host,'path':path,'destination':str(dst.relative_to(DEST)),'returncode':r.returncode,'stderr':r.stderr[-600:]}

def main():
    queue=list(ENTRIES); done=set()
    while queue:
        name=queue.pop()
        if name in done: continue
        p=PPC/(name+'.py')
        if not p.exists(): continue
        done.add(name);copy(p,ARCHIVE/'code'/p.name)
        for node in ast.walk(ast.parse(p.read_text())):
            imports=([node.module] if isinstance(node,ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            queue.extend(n.split('.')[0] for n in imports if (PPC/(n.split('.')[0]+'.py')).exists())
    # Keep the original sibling layout expected by the custom NR importer.
    for p in (SRC/'ScenarioSynthesis_PPC').glob('*.py'):
        if not p.name.startswith(('cleanup','delete')): copy(p,ARCHIVE/'ScenarioSynthesis_PPC'/p.name)
    for p in PPC.glob('*.md'):
        if p.name in {'CONTROLLED_ERROR_GEOMETRY_PROTOCOL.md','PF_V2_FOUR_MODEL_GUIDE.md','PF_SURROGATE_USAGE.md','PIGNN_INFERENCE_GUIDE.md'}:
            copy(p,DEST/'docs'/'historical'/p.name)
    for group in GROUPS+ALEX_GROUPS:
        tree(PPC/'results'/group,ARCHIVE/'results'/'local'/group)
    tokens=('csp','nr1','geometry','basis','shrinkage','jacobian','conditional','inference','score_graphkit_e120','score_lumina_gbv06z','model_seed','pfv2_gridsfm','pignn_global','gbcorr')
    for p in (PPC/'sbatch').rglob('*'):
        if p.is_file() and p.suffix in {'.sh','.sbatch','.tsv'} and any(t in p.name for t in tokens): copy(p,ARCHIVE/'experiments'/'historical_local'/p.relative_to(PPC/'sbatch'))
    # Preserve the supplied final manuscript without overwriting the user's draft.
    copy(Path('/Users/changhunkim/.codex/attachments/55d341e0-0172-4269-bff8-e2aa03b1a2d8/pasted-text.txt'),DEST/'paper'/'submitted_manuscript.tex')
    for p in (DEST/'ICASSP2027_manuscript').iterdir():
        if p.is_file() and (p.suffix in {'.bib','.bst','.sty'} or p.name in {'csp_diagram_finished_big.pdf','figure1_controlled_error_geometry_larger_text_final_v9_recreated.pdf','gridsfm_raw_vs_csp16_discrete3_larger_text_v7.pdf'}):copy(p,DEST/'paper'/p.name)
    reports=['dataset_ppnr-v2_31-grid-overview','acpf_ppnr-v2_four-model-results','acpf_controlled-error-geometry_gbnetwork','acpf_gbnetwork_case300_global-context-report','acpf_gridfm-graphkit_model-formulation','acpf_gridsfm_model-formulation','acpf_lumina-2m_model-formulation']
    for p in (PPC/'latex').rglob('*'):
        if p.is_file() and p.stem in reports and p.suffix in {'.tex','.pdf'}:copy(p,ARCHIVE/'reports'/p.relative_to(PPC/'latex'))
    log=[]
    for group in GROUPS:
        log.append(remote('helma',REMOTE+'/PIGNN-Attn-LS-PPC/results/'+group,ARCHIVE/'results'/'helma'/group))
        log.append(remote('helma',REMOTE+'/PIGNN-Attn-LS-PPC/overlays/'+group,ARCHIVE/'experiments'/'snapshots'/'helma'/group,{'.py','.sh','.json','.tsv','.md'}))
    for group in ALEX_GROUPS:
        log.append(remote('alex',ALEX+'/PIGNN-Attn-LS-PPC/results/'+group,ARCHIVE/'results'/'alex'/group))
        log.append(remote('alex',ALEX+'/PIGNN-Attn-LS-PPC/overlays/'+group,ARCHIVE/'experiments'/'snapshots'/'alex'/group,{'.py','.sh','.json','.tsv','.md'}))
    log.append(remote('helma',REMOTE+'/PIGNN-Attn-LS-PPC/sbatch',ARCHIVE/'experiments'/'historical_helma',{'.sh','.sbatch','.tsv'}))
    for name in ('GridSFM','LUMINA','gridfm-graphkit','lumina-sdk'):
        log.append(remote('helma',REMOTE+'/'+name,DEST/'third_party'/name,{'.py','.json','.yaml','.yml','.toml','.txt','.md','.cfg','.ini'}))
    out=ARCHIVE/'provenance';out.mkdir(exist_ok=True)
    (out/'local_copy_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    (out/'remote_transfer_log.json').write_text(json.dumps(log,indent=2)+'\n')
    print(f'Copied {len(records)} local files; {len(log)} remote source transfers.')

if __name__=='__main__':main()
