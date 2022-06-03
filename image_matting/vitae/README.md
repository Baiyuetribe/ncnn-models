# Rethinking Portrait Matting with Privacy Preserving

## Input --> Output

![](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting/raw/main/demo/gif/4.gif)

## Convert 


在onnx2ncnn步骤失败，报错如下:
```ruby
onnx转换成功，但onnx-sim失败，报错如下：
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Fatal error: ATen is not a registered function/op
```

```ruby
pnnx转换后可生成文件，但ncnn转换失败，报错如下：
pnnxparam = ts.pnnx.param
pnnxbin = ts.pnnx.bin
pnnxpy = ts_pnnx.py
ncnnparam = ts.ncnn.param
ncnnbin = ts.ncnn.bin
ncnnpy = ts_ncnn.py
optlevel = 2
device = cpu
inputshape = [1,3,512,512]f32
inputshape2 =
customop =
moduleop =
############# pass_level0
inline module = network.ViTAE_S.NormalCell.Attention
inline module = network.ViTAE_S.NormalCell.AttentionPerformer
inline module = network.ViTAE_S.NormalCell.Mlp
inline module = network.ViTAE_S.NormalCell.NormalCell
inline module = network.ViTAE_S.base_model.BasicLayer
inline module = network.ViTAE_S.base_model.PatchEmbed
inline module = network.ViTAE_S.base_model.PatchMerging
inline module = network.ViTAE_S.base_model.ViTAE_noRC_MaxPooling_basic
inline module = network.ViTAE_S.models.ViTAE_noRC_MaxPooling_DecoderV1
inline module = network.modules.DBFI
inline module = network.modules.SBFI
inline module = network.modules.TFI
inline module = torch.nn.modules.linear.Identity
inline module = network.ViTAE_S.NormalCell.Attention
inline module = network.ViTAE_S.NormalCell.AttentionPerformer
inline module = network.ViTAE_S.NormalCell.Mlp
inline module = network.ViTAE_S.NormalCell.NormalCell
inline module = network.ViTAE_S.base_model.BasicLayer
inline module = network.ViTAE_S.base_model.PatchEmbed
inline module = network.ViTAE_S.base_model.PatchMerging
inline module = network.ViTAE_S.base_model.ViTAE_noRC_MaxPooling_basic
inline module = network.ViTAE_S.models.ViTAE_noRC_MaxPooling_DecoderV1
inline module = network.modules.DBFI
inline module = network.modules.SBFI
inline module = network.modules.TFI
inline module = torch.nn.modules.linear.Identity
60  62  B.3  80  82  84  101  102  104  105  106  107  109  110  111  112  input.39  114  116  117  118  119  120  121  122  wh.2  3030  ww.2  129  b.3  n.3  c.3  153  155  input.41  165  166  167  168  169  170  171  172  174  175  177  convX.3  180  181  185  188  190  B.5  N.3  213  215  kqv0.2  x.7  x1.1  v.3  221  223  225  xd.2  227  228  wtx.2  231  232  kp.2  234  236  238  3070  xd0.2  240  241  wtx0.2  244  245  3084  246  248  250  D.2  252  254  256  258  259  y.2  262  input.45  265  266  3103  267  x.9  input0.33  271  276  277  278  279  280  3106  281  x0.28  285  b.5  n.5  c.5  309  311  input.47  321  322  323  324  325  326  327  328  330  331  333  convX.5  336  337  341  344  346  B.7  N.5  369  371  kqv0.1  x.24  x0.24  v.5  377  379  381  xd.1  383  384  wtx.1  387  388  kp.1  390  392  394  3146  xd0.1  396  397  wtx0.1  400  401  3160  402  404  406  D.1  408  410  412  414  415  y.1  418  input.15  421  422  3179  423  x.3  input0.29  427  432  433  434  435  436  3182  437  x0.3  442  443  444  445  446  447  448  H.1  3184  W.1  454  B.9  C.3  476  478  480  input.17  483  484  486  488  input0.31  490  492  493  494  wh.4  3194  ww.4  501  b.7  n.7  c.7  525  527  input.19  537  538  539  540  541  542  543  544  546  547  549  convX.7  552  554  556  B.11  N.7  C.5  579  580  583  qkv0.3  q.3  k.3  v.7  589  590  attn.9  input.21  593  594  595  input1.2  598  599  3219  600  x.26  input0.35  604  609  610  611  612  613  3222  614  x0.26  618  b.9  n.9  c.9  642  644  input.23  654  655  656  657  658  659  660  661  663  664  666  convX.9  669  671  673  B.13  N.9  C.7  696  697  700  qkv0.5  q.5  k.5  v.9  706  707  attn.13  input.25  710  711  712  input0.37  715  716  3247  717  x.30  input0.39  721  726  727  728  729  730  3250  731  x0.30  736  737  738  H0.1  3253  W0.1  744  B.15  C.9  786  788  790  input.27  793  794  796  798  input0.41  800  802  803  804  wh.6  3263  ww.6  831  b.11  n.11  c.11  855  857  input.29  867  868  869  870  871  872  873  874  876  877  879  convX.11  882  884  886  B.17  N.11  C.11  909  910  913  qkv0.7  q.7  k.7  v.11  919  920  attn.17  input.31  923  924  925  input0.19  928  929  3288  930  x.32  input0.21  934  939  940  941  942  943  3291  944  x0.32  948  b.4  n.4  c.4  972  974  input.8  984  985  986  987  988  989  990  991  993  994  996  convX.4  999  1001  1003  B.6  N.4  C.6  1026  1027  1030  qkv0.4  q.4  k.4  v.4  1036  1037  attn.10  input.10  1040  1041  1042  input0.8  1045  1046  3316  1047  x.4  input0.10  1051  1056  1057  1058  1059  1060  3319  1061  x0.4  1065  b.6  n.6  c.6  1089  1091  input.12  1101  1102  1103  1104  1105  1106  1107  1108  1110  1111  1113  convX.6  1116  1118  1120  B.8  N.6  C.8  1143  1144  1147  qkv0.6  q.6  k.6  v.6  1153  1154  attn.14  input.14  1157  1158  1159  input0.12  1162  1163  3344  1164  x.6  input0.14  1168  1173  1174  1175  1176  1177  3347  1178  x0.6  1182  b.8  n.8  c.8  1206  1208  input.16  1218  1219  1220  1221  1222  1223  1224  1225  1227  1228  1230  convX.8  1233  1235  1237  B.10  N.8  C.10  1260  1261  1264  qkv0.8  q.8  k.8  v.8  1270  1271  attn.18  input.18  1274  1275  1276  input0.16  1279  1280  3372  1281  x.8  input0.18  1285  1290  1291  1292  1293  1294  3375  1295  x0.8  1299  b.10  n.10  c.10  1323  1325  input.20  1335  1336  1337  1338  1339  1340  1341  1342  1344  1345  1347  convX.10  1350  1352  1354  B.12  N.10  C.12  1377  1378  1381  qkv0.10  q.10  k.10  v.10  1387  1388  attn.22  input.22  1391  1392  1393  input0.20  1396  1397  3400  1398  x.10  input0.22  1402  1407  1408  1409  1410  1411  3403  1412  x0.10  1416  b.12  n.12  c.12  1440  1442  input.24  1452  1453  1454  1455  1456  1457  1458  1459  1461  1462  1464  convX.12  1467  1469  1471  B.14  N.12  C.14  1494  1495  1498  qkv0.12  q.12  k.12  v.12  1504  1505  attn.26  input.26  1508  1509  1510  input0.24  1513  1514  3428  1515  x.12  input0.26  1519  1524  1525  1526  1527  1528  3431  1529  x0.12  1533  b.14  n.14  c.14  1557  1559  input.28  1569  1570  1571  1572  1573  1574  1575  1576  1578  1579  1581  convX.14  1584  1586  1588  B.16  N.14  C.16  1611  1612  1615  qkv0.14  q.14  k.14  v.14  1621  1622  attn.30  input.30  1625  1626  1627  input0.28  1630  1631  3456  1632  x.14  input0.30  1636  1641  1642  1643  1644  1645  3459  1646  x0.14  1650  b.16  n.16  c.16  1674  1676  input.32  1686  1687  1688  1689  1690  1691  1692  1693  1695  1696  1698  convX.16  1701  1703  1705  B.18  N.16  C.18  1728  1729  1732  qkv0.16  q.16  k.16  v.16  1738  1739  attn.34  input.34  1742  1743  1744  input0.32  1747  1748  3484  1749  x.16  input0.34  1753  1758  1759  1760  1761  1762  3487  1763  x0.16  1767  b.18  n.18  c.18  1791  1793  input.36  1803  1804  1805  1806  1807  1808  1809  1810  1812  1813  1815  convX.18  1818  1820  1822  B.20  N.18  C.20  1845  1846  1849  qkv0.18  q.18  k.18  v.18  1855  1856  attn.38  input.38  1859  1860  1861  input0.36  1864  1865  3512  1866  x.18  input0.38  1870  1875  1876  1877  1878  1879  3515  1880  x0.18  1884  b.20  n.20  c.20  1908  1910  input.40  1920  1921  1922  1923  1924  1925  1926  1927  1929  1930  1932  convX.20  1935  1937  1939  B.22  N.20  C.22  1962  1963  1966  qkv0.20  q.20  k.20  v.20  1972  1973  attn.42  input.42  1976  1977  1978  input0.40  1981  1982  3540  1983  x.20  input0.42  1987  1992  1993  1994  1995  1996  3543  1997  x0.20  2001  b.22  n.22  c.22  2025  2027  input.44  2037  2038  2039  2040  2041  2042  2043  2044  2046  2047  2049  convX.22  2052  2054  2056  B.24  N.22  C.24  2079  2080  2083  qkv0.22  q.22  k.22  v.22  2089  2090  attn.46  input.46  2093  2094  2095  input0.44  2098  2099  3568  2100  x.22  input0.46  2104  2109  2110  2111  2112  2113  3571  2114  x0.22  2118  b.13  n.13  c.13  2142  2144  input.33  2154  2155  2156  2157  2158  2159  2160  2161  2163  2164  2166  convX.13  2169  2171  2173  B.19  N.13  C.13  2196  2197  2200  qkv0.9  q.9  k.9  v.13  2206  2207  attn.21  input.35  2210  2211  2212  input0.23  2215  2216  3596
----------------

inline module = network.ViTAE_S.NormalCell.Attention
inline module = network.ViTAE_S.NormalCell.AttentionPerformer
inline module = network.ViTAE_S.NormalCell.Mlp
inline module = network.ViTAE_S.NormalCell.NormalCell
inline module = network.ViTAE_S.base_model.BasicLayer
inline module = network.ViTAE_S.base_model.PatchEmbed
inline module = network.ViTAE_S.base_model.PatchMerging
inline module = network.ViTAE_S.base_model.ViTAE_noRC_MaxPooling_basic
inline module = network.ViTAE_S.models.ViTAE_noRC_MaxPooling_DecoderV1
inline module = network.modules.DBFI
inline module = network.modules.SBFI
inline module = network.modules.TFI
inline module = torch.nn.modules.linear.Identity
2217  x.34  input0.25  2221  2226  2227  2228  2229  2230  3599  2231  x0.34  2236  2237  2238  3601  H1.1  3603  W1.1  2244  B.2  C.2  2266  2268  2270  input.2  2273  2274  2276  2278  input0.2  2280  2282  2283  2284  wh.1  3613  ww.1  2291  b.2  n.2  c.2  2315  2317  input.4  2327  2328  2329  2330  2331  2332  2333  2334  2336  2337  2339  convX.2  2342  2344  2346  B.4  N.2  C.4  2370  2371  2374  qkv0.2  q.2  k.2  v.2  2380  2381  attn.6  input.6  2384  2385  2386  input0.4  2389  2390  3637  2391  x.2  input0.6  2395  2400  2401  2402  2403  2404  3640  2405  x0.2  2409  b.1  n.1  c.1  2433  2435  input.43  2445  2446  2447  2448  2449  2450  2451  2452  2454  2455  2457  convX.1  2460  2462  2464  B.1  N.1  C.1  2488  2489  2492  qkv0.1  q.1  k.1  v.1  2498  2499  attn.1  input.49  2502  2503  2504  input0.43  2507  2508  3664  2509  x.1  input0.27  2513  2518  2519  2520  2521  2522  3667  2523  x0.1  2528  2529  2530  3669  2531  3671  2533  2536  2538  input.37  9  10  11  12  13  14  15  16  17  18  19  20  21  2541  2542  2545  2546  2547  2586  2587  2588  2589  2590  2591  2592  2593  2594  2595  2606  2607  2608  2609  2610  2611  2612  2613  2614  2615  2625  2626  2627  input.5  2630  2631  2632  input2.1  2634  2635  2637  2638  2649  2650  2651  2652  2653  2654  2655  2656  2657  2658  2668  2669  2670  input.7  2673  2674  2675  input0.5  2677  2678  2680  2681  2692  2693  2694  2695  2696  2697  2698  2699  2700  2701  2711  2712  2713  input.9  2716  2717  2718  input0.7  2720  2721  2723  2724  2733  2734  2735  2736  2737  2738  2739  2740  2741  2751  2752  2753  2754  2755  2756  2757  2758  2759  2761  2763  2764  2765  3680  2766  2767  3683  2769  3685  2770  3686  2771  3688  2772  input.3  2781  2782  2783  input0.9  2786  2787  2788  2798  2799  2800  2801  2802  2803  2804  2805  2806  2808  2810  3692  2811  3694  2812  3695  2813  3697  2814  3699  2816  3701  2817  3702  2818  3704  2819  input0.3  2828  2829  2830  input0.11  2833  2834  2835  2843  2844  2845  input.11  2848  2849  2850  input1.1  2861  2862  2863  2864  2865  2866  2867  2868  2869  2871  2873  3709  2874  3711  2875  3712  2876  3714  2877  3716  2879  3718  2880  3719  2881  3721  2882  input1.3  2891  2892  2893  input0.13  2896  2897  2898  2906  2907  2908  input.13  2911  2912  2913  input0.15  2924  2925  2926  2927  2928  2929  2930  2931  2932  2934  2936  3726  2937  3728  2938  3729  2939  3731  2940  3733  2942  3735  2943  3736  2944  3738  2945  input2.3  2954  2955  2956  input0.17  2959  2960  2961  2969  2970  2971  input.1  2974  2975  2976  input0.1  2984  2985  2986  2987  2988  2989  2991  2993  3743  2994  3745  2995  3746  2996  3748  2997  3750  2999  3752  3000  3753  3001  3755  3002  input3.1  3006  local_sigmoid.1  3008  index.1  3010  3011  3012  3013  index0.1  trimap_mask.1  3016  trimap_mask0.1  fg_mask.1  3020  3776  fg_mask0.1  3023  fg_mask1.1  3026  3027  38  39  40  41  42  43
----------------

foldable_constant 228
foldable_constant 241
foldable_constant 397
foldable_constant 384
############# pass_level1
no attribute value
unknown Parameter value kind prim::Constant
unknown Parameter value kind prim::Constant of TensorType, t.dim = 3
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
unknown Parameter value kind prim::Constant
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
unknown Parameter value kind prim::Constant
unknown Parameter value kind prim::Constant of TensorType, t.dim = 3
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
unknown Parameter value kind prim::Constant
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
no attribute value
############# pass_level2
############# pass_level3
assign unique operator name pnnx_unique_0 to encoder.layers.0.NC.0.mlp.drop
assign unique operator name pnnx_unique_1 to encoder.layers.0.NC.1.mlp.drop
assign unique operator name pnnx_unique_2 to encoder.layers.1.NC.0.mlp.drop
assign unique operator name pnnx_unique_3 to encoder.layers.1.NC.1.mlp.drop
assign unique operator name pnnx_unique_4 to encoder.layers.2.NC.0.mlp.drop
assign unique operator name pnnx_unique_5 to encoder.layers.2.NC.1.mlp.drop
assign unique operator name pnnx_unique_6 to encoder.layers.2.NC.2.mlp.drop
assign unique operator name pnnx_unique_7 to encoder.layers.2.NC.3.mlp.drop
assign unique operator name pnnx_unique_8 to encoder.layers.2.NC.4.mlp.drop
assign unique operator name pnnx_unique_9 to encoder.layers.2.NC.5.mlp.drop
assign unique operator name pnnx_unique_10 to encoder.layers.2.NC.6.mlp.drop
assign unique operator name pnnx_unique_11 to encoder.layers.2.NC.7.mlp.drop
assign unique operator name pnnx_unique_12 to encoder.layers.2.NC.8.mlp.drop
assign unique operator name pnnx_unique_13 to encoder.layers.2.NC.9.mlp.drop
assign unique operator name pnnx_unique_14 to encoder.layers.2.NC.10.mlp.drop
assign unique operator name pnnx_unique_15 to encoder.layers.2.NC.11.mlp.drop
assign unique operator name pnnx_unique_16 to encoder.layers.3.NC.0.mlp.drop
assign unique operator name pnnx_unique_17 to encoder.layers.3.NC.1.mlp.drop
assign unique operator name pnnx_unique_18 to decoder.tfi_3.transform
assign unique operator name pnnx_unique_19 to decoder.tfi_3.transform
assign unique operator name pnnx_unique_20 to decoder.tfi_2.transform
assign unique operator name pnnx_unique_21 to decoder.tfi_2.transform
assign unique operator name pnnx_unique_22 to decoder.tfi_1.transform
assign unique operator name pnnx_unique_23 to decoder.tfi_1.transform
assign unique operator name pnnx_unique_24 to decoder.tfi_0.transform
assign unique operator name pnnx_unique_25 to decoder.tfi_0.transform
eliminate_noop_math aten::sub pnnx_2478
eliminate_noop_math aten::sub pnnx_2466
eliminate_noop_math aten::sub pnnx_2438
eliminate_noop_math aten::sub pnnx_2426
eliminate_noop_math aten::sub pnnx_2398
eliminate_noop_math aten::sub pnnx_2386
eliminate_noop_math aten::sub pnnx_2358
eliminate_noop_math aten::sub pnnx_2346
eliminate_noop_math aten::sub pnnx_2323
eliminate_noop_math aten::sub pnnx_2311
eliminate_noop_math aten::mul pnnx_2241
eliminate_noop_math aten::mul pnnx_2234
eliminate_noop_math aten::mul pnnx_2178
eliminate_noop_math aten::mul pnnx_2138
eliminate_noop_math aten::mul pnnx_2131
eliminate_noop_math aten::mul pnnx_2075
eliminate_noop_math aten::mul pnnx_1981
eliminate_noop_math aten::mul pnnx_1974
eliminate_noop_math aten::mul pnnx_1918
eliminate_noop_math aten::mul pnnx_1878
eliminate_noop_math aten::mul pnnx_1871
eliminate_noop_math aten::mul pnnx_1815
eliminate_noop_math aten::mul pnnx_1775
eliminate_noop_math aten::mul pnnx_1768
eliminate_noop_math aten::mul pnnx_1712
eliminate_noop_math aten::mul pnnx_1672
eliminate_noop_math aten::mul pnnx_1665
eliminate_noop_math aten::mul pnnx_1609
eliminate_noop_math aten::mul pnnx_1569
eliminate_noop_math aten::mul pnnx_1562
eliminate_noop_math aten::mul pnnx_1506
eliminate_noop_math aten::mul pnnx_1466
eliminate_noop_math aten::mul pnnx_1459
eliminate_noop_math aten::mul pnnx_1403
eliminate_noop_math aten::mul pnnx_1363
eliminate_noop_math aten::mul pnnx_1356
eliminate_noop_math aten::mul pnnx_1300
eliminate_noop_math aten::mul pnnx_1260
eliminate_noop_math aten::mul pnnx_1253
eliminate_noop_math aten::mul pnnx_1197
eliminate_noop_math aten::mul pnnx_1157
eliminate_noop_math aten::mul pnnx_1150
eliminate_noop_math aten::mul pnnx_1094
eliminate_noop_math aten::mul pnnx_1054
eliminate_noop_math aten::mul pnnx_1047
eliminate_noop_math aten::mul pnnx_991
eliminate_noop_math aten::mul pnnx_951
eliminate_noop_math aten::mul pnnx_944
eliminate_noop_math aten::mul pnnx_888
eliminate_noop_math aten::mul pnnx_848
eliminate_noop_math aten::mul pnnx_841
eliminate_noop_math aten::mul pnnx_785
eliminate_noop_math aten::mul pnnx_672
eliminate_noop_math aten::mul pnnx_665
eliminate_noop_math aten::mul pnnx_609
eliminate_noop_math aten::mul pnnx_569
eliminate_noop_math aten::mul pnnx_562
eliminate_noop_math aten::mul pnnx_506
eliminate_noop_math aten::mul pnnx_414
eliminate_noop_math aten::mul pnnx_407
eliminate_noop_math aten::mul pnnx_262
eliminate_noop_math aten::mul pnnx_222
eliminate_noop_math aten::mul pnnx_215
eliminate_noop_math aten::mul pnnx_70
############# pass_level4
############# pass_level5
############# pass_ncnn
insert_reshape_pooling 4
insert_reshape_pooling 4
insert_reshape_pooling 4
insert_reshape_pooling 4
insert_reshape_pooling 4
insert_reshape_pooling 4
insert_reshape_pooling 4
insert_reshape_pooling 4
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index -1080805153 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index -1 is not supported yet!
reshape tensor with batch index -2075006752 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 64 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index -1084731127 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1053579732 is not supported yet!
reshape tensor with batch index 1051205988 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 64 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index -2075006496 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 256 is not supported yet!
reshape tensor with batch index 256 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1024 is not supported yet!
reshape tensor with batch index 1024 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 1061049605 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index -2009340304 is not supported yet!
reshape tensor with batch index -2009340304 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index -1811873792 is not supported yet!
reshape tensor with batch index 32 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index -2075006496 is not supported yet!
reshape tensor with batch index 1 is not supported yet!
reshape tensor with batch index 256 is not supported yet!
reshape tensor with batch index 2 is not supported yet!
reshape tensor with batch index 32 is not supported yet!
reshape tensor with batch index 1058817295 is not supported yet!
reshape tensor with batch index 2 is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
permute across batch dim is not supported yet!
ignore nn.MaxPool2d encoder.layers.0.RC.maxpool1 param ceil_mode=False
ignore nn.MaxPool2d encoder.layers.0.RC.maxpool1 param dilation=(1,1)
ignore nn.MaxPool2d encoder.layers.0.RC.maxpool1 param kernel_size=(2,2)
ignore nn.MaxPool2d encoder.layers.0.RC.maxpool1 param padding=(0,0)
ignore nn.MaxPool2d encoder.layers.0.RC.maxpool1 param return_indices=True
ignore nn.MaxPool2d encoder.layers.0.RC.maxpool1 param stride=(2,2)
ignore nn.MaxPool2d encoder.layers.0.RC.maxpool2 param ceil_mode=False
ignore nn.MaxPool2d encoder.layers.0.RC.maxpool2 param dilation=(1,1)
ignore nn.MaxPool2d encoder.layers.0.RC.maxpool2 param kernel_size=(2,2)
ignore nn.MaxPool2d encoder.layers.0.RC.maxpool2 param padding=(0,0)
ignore nn.MaxPool2d encoder.layers.0.RC.maxpool2 param return_indices=True
ignore nn.MaxPool2d encoder.layers.0.RC.maxpool2 param stride=(2,2)
ignore nn.MaxPool2d encoder.layers.1.RC.pooling param ceil_mode=False
ignore nn.MaxPool2d encoder.layers.1.RC.pooling param dilation=(1,1)
ignore nn.MaxPool2d encoder.layers.1.RC.pooling param kernel_size=(2,2)
ignore nn.MaxPool2d encoder.layers.1.RC.pooling param padding=(0,0)
ignore nn.MaxPool2d encoder.layers.1.RC.pooling param return_indices=True
ignore nn.MaxPool2d encoder.layers.1.RC.pooling param stride=(2,2)
ignore nn.MaxPool2d encoder.layers.2.RC.pooling param ceil_mode=False
ignore nn.MaxPool2d encoder.layers.2.RC.pooling param dilation=(1,1)
ignore nn.MaxPool2d encoder.layers.2.RC.pooling param kernel_size=(2,2)
ignore nn.MaxPool2d encoder.layers.2.RC.pooling param padding=(0,0)
ignore nn.MaxPool2d encoder.layers.2.RC.pooling param return_indices=True
ignore nn.MaxPool2d encoder.layers.2.RC.pooling param stride=(2,2)
ignore nn.MaxPool2d encoder.layers.3.RC.pooling param ceil_mode=False
ignore nn.MaxPool2d encoder.layers.3.RC.pooling param dilation=(1,1)
ignore nn.MaxPool2d encoder.layers.3.RC.pooling param kernel_size=(2,2)
ignore nn.MaxPool2d encoder.layers.3.RC.pooling param padding=(0,0)
ignore nn.MaxPool2d encoder.layers.3.RC.pooling param return_indices=True
ignore nn.MaxPool2d encoder.layers.3.RC.pooling param stride=(2,2)
ignore pnnx.Expression pnnx_expr_180 param expr=1.000000e+00
ignore pnnx.Expression pnnx_expr_179 param expr=0.000000e+00
ignore pnnx.Expression pnnx_expr_141 param expr=[32,32]
ignore pnnx.Expression pnnx_expr_115 param expr=[64,64]
ignore pnnx.Expression pnnx_expr_86 param expr=[128,128]
ignore pnnx.Expression pnnx_expr_57 param expr=[256,256]
ignore pnnx.Expression pnnx_expr_28 param expr=[512,512]
ignore torch.eq torch.eq_91 param other=2
ignore pnnx.Expression pnnx_expr_11 param expr=[@0]
ignore pnnx.Expression pnnx_expr_10 param expr=False
ignore torch.eq torch.eq_92 param other=1
ignore pnnx.Expression pnnx_expr_8 param expr=[@0]
ignore torch.eq torch.eq_93 param other=2
ignore pnnx.Expression pnnx_expr_4 param expr=[@0]
```


## Reference

- [ViTAE-Transformer/ViTAE-Transformer-Matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting)


