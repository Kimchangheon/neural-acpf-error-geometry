# ICASSP 2027 참고문헌·인용 점검 결과

## 범위

이번 수정은 인용의 적절성, 관련 연구 포지셔닝, BibTeX 메타데이터만 다뤘다. 수학적 정의, 실험 수치, claim 강도, 표·그림 구성 등 리뷰의 다른 항목은 수정하지 않았다.

## 원고에서 수정한 인용

1. **직접 PF 회귀와 AC-OPF 모델을 구분**했다. 직접 PF 회귀의 근거로 PowerFlowNet을 사용하고, CANOS·HH-MPNN은 scalable/cross-grid AC-OPF 계열로 이동했다.
2. **feasibility/correction taxonomy**를 구체화했다.
   - PF-based completion: DeepOPF
   - differentiable completion/correction: DC3
   - iterative/feasibility-seeking correction: PIGNN-Attn-LS, FSNet
   - learned nonlinear approximate projection: FAB
   CSP는 이들과 달리 training solutions로 적합한 선형 post-training map이며 exact feasibility를 보장하지 않는다고 구분했다.
3. **Goodwin et al.의 formal PF-manifold geometry**를 Introduction과 method §2.2에 모두 추가하고, 본 논문의 empirical linear error geometry와 구분했다.
4. **Park et al.을 가까운 선행연구로 더 정확히 설명**했다. PCA 압축뿐 아니라 principal-component space에서 학습, full-output reconstruction, exact AC solver warm start까지 적었다. CSP와의 차이는 frozen surrogate에 대한 post-hoc diagnostic/correction, 별도 bias calibration, fixed-norm intervention으로 정리했다.
5. **benchmark/data references**로 PFΔ, PGLib-OPF, OPFData를 추가했다. 이들은 현재 GBnetwork 데이터의 provenance로 주장하지 않고, open benchmark resources로만 인용했다.
6. **software/numerical references**로 pandapower, MATPOWER, Tinney--Hart Newton power flow를 추가했다.
7. **PIGNN-GC citation scope**를 바로잡았다. `kim2026pignn`이 PIGNN-GC 자체를 정의하는 것처럼 쓰지 않고, “our global-context extension of PIGNN-Attn-LS”로 표현했다.
8. **GridFM-GraphKit software citation**을 재현 가능한 versioned citation으로 바꿨다: v0.9.0 PyPI release URL과 source SHA-256을 포함했다.
9. **PowerFlowNet 미포함 문제**에는 실험이 comprehensive learned-solver benchmark가 아니라 selected scratch-trained outputs에 대한 post-processing study라는 범위 문장을 추가했다.
10. topology-shift의 큰 maximum residual 해석에는 worst-case verification 연구를 인용했다.

## BibTeX 메타데이터에서 수정·추가한 주요 항목

- `rivera2025pfdelta`: 저자명을 **Ana Rivera Him**으로 수정하고 NeurIPS 2025 Datasets and Benchmarks Track, DOI, proceedings URL을 기록했다.
- `nguyen2025fsnet`: NeurIPS 38, pp. 44366--44404, DOI 및 proceedings URL을 추가했다.
- `gridfmgraphkit2026`: v0.9.0, release date, versioned PyPI URL, source SHA-256을 추가했다.
- `puech2026genco`: 공식 제목을 `GENCO---A Unified ...`로 정리하고 arXiv DOI/URL을 추가했다.
- `conrad2025data`: 올바른 arXiv ID `2602.19667`과 URL을 기록했다.
- `donti2021dc3`, `pan2020deepopf`, `chzhen2026fab`, `chevalier2022guarantees`: arXiv DOI와 URL을 추가했다.
- `goodwin2026geometry`, `nellikkath2022pinn`, `thurner2018pandapower`, `babaeinejadsarookolaee2021pglib`를 새로 추가했다.
- `park2024compact`, `lin2024powerflownet`, `arowolo2026hhmpnn`, `wen2026pigat`, `yang2026gridsfm`, `li2026lumina` 등 기존 핵심 항목의 title/author/venue/DOI 또는 official URL을 재점검했다.

## 자동 일관성 검사

- revised `.tex`의 고유 citation key: **29개**
- revised `.bib`의 entry: **29개**
- 누락 key: **0개**
- 중복 key: **0개**
- 인용되지 않는 entry: **0개**
- `bibtexu`를 이용한 BibTeX parse: **통과**

정리된 `.bib`에는 revised manuscript에서 실제로 인용되는 entry만 남겼다. 원래 `.bib`의 미사용 entry는 원본 파일에 그대로 보존되어 있다.

## 저자 확인이 필요한 잔여 항목

1. **PIGNN-GC에서 PIGNN-Attn-LS의 line-search operator가 활성화되어 있는지**는 제공된 파일만으로 확인되지 않는다. revised manuscript는 citation 범위만 바로잡았으며, 최종본에는 retained/disabled 여부를 한 구절로 추가하는 것이 좋다.
2. manuscript가 실제로 사용한 **pandapower의 정확한 버전**은 제공되지 않았다. 논문 또는 repository manifest에 version을 추가해야 한다.
3. **31개 cross-grid case의 정확한 case identifier와 source/version**은 제공되지 않았다. revised text는 pandapower/PYPOWER representation만 인용하며, case provenance를 임의로 만들지 않았다.
4. **GridFM-GraphKit v0.9.0**이 실제 최종 실험 환경의 package version인지 확인해야 한다. 현재 `.bib` 및 원고에 주어진 v0.9.0을 기준으로 고정했다.
5. full manuscript compile은 `spconf.sty`, `IEEEbib.bst`, 일부 figure asset이 업로드되지 않아 수행하지 않았다. 다만 cite-key 및 BibTeX syntax 검사는 통과했다.
