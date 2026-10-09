# ppNR v2 corpus: selected by the _u0clean_sgenpert_pvqvstart_ configuration
# tag, which is emitted only when --rand_u_start is absent, --perturb_sgen
# and --pv_q_from_vstart both hold. The _ppNR_ solver tag alone no longer
# identifies a generation -- several now share it.
# grid|buses|file|bytes, ascending by bus count.
MANIFEST=(
  "case4gs|4|case4gs_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|49820528"   # 47 MiB
  "case5|5|case5_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|55716060"   # 53 MiB
  "case6ww|6|case6ww_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|64073509"   # 61 MiB
  "case9|9|case9_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|68836125"   # 65 MiB
  "case14|14|case14_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|97615718"   # 93 MiB
  "case24_ieee_rts|24|case24_ieee_rts_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|146697014"   # 139 MiB
  "GBreducednetwork|29|GBreducednetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|233349421"   # 222 MiB
  "case30|30|case30_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|164003079"   # 156 MiB
  "case_ieee30|30|case_ieee30_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|166082420"   # 158 MiB
  "case33bw|33|case33bw_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|167211135"   # 159 MiB
  "case39|39|case39_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|196206525"   # 187 MiB
  "case57|57|case57_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37034_NR_branchrows_directSI.parquet|287509489"   # 274 MiB
  "case89pegase|89|case89pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_36999_NR_branchrows_directSI.parquet|509673974"   # 486 MiB
  "SimBench|94|SimBench_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_38649_NR_branchrows_directSI.parquet|448628631"   # 427 MiB
  "case118|118|case118_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|595248145"   # 567 MiB
  "case145|145|case145_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_36918_NR_branchrows_directSI.parquet|611204073"   # 582 MiB
  "iceland|189|iceland_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37112_NR_branchrows_directSI.parquet|757733171"   # 722 MiB
  "case_illinois200|200|case_illinois200_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|878393968"   # 837 MiB
  "case300|300|case300_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37930_NR_branchrows_directSI.parquet|895340949"   # 853 MiB
  "LVN_heo1|722|LVN_heo1_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|1511991796"   # 1441 MiB
  "case1354pegase|1354|case1354pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37038_NR_branchrows_directSI.parquet|3079212321"   # 2936 MiB
  "case1888rte|1888|case1888rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|4292592706"   # 4093 MiB
  "GBnetwork|2224|GBnetwork_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37022_NR_branchrows_directSI.parquet|4614954424"   # 4401 MiB
  "case2848rte|2848|case2848rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|6429099187"   # 6131 MiB
  "case2869pegase|2869|case2869pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37099_NR_branchrows_directSI.parquet|6574437865"   # 6269 MiB
  "case3120sp|3120|case3120sp_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37191_NR_branchrows_directSI.parquet|7102809855"   # 6773 MiB
  "ENTSO_E_RealGridTest|6051|ENTSO_E_RealGridTest_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37000_NR_branchrows_directSI.parquet|13306582042"   # 12690 MiB
  "case6470rte|6470|case6470rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37895_NR_branchrows_directSI.parquet|14691691336"   # 14011 MiB
  "case6495rte|6495|case6495rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37889_NR_branchrows_directSI.parquet|14749538149"   # 14066 MiB
  "case6515rte|6515|case6515rte_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_37211_NR_branchrows_directSI.parquet|14515792682"   # 13843 MiB
  "case9241pegase|9241|case9241pegase_ppcY_backbone_dc_compile_ppNR_ls0.60-1.40_u0clean_sgenpert_pvqvstart_siNR_38508_NR_branchrows_directSI.parquet|22146434116"   # 21120 MiB
)
