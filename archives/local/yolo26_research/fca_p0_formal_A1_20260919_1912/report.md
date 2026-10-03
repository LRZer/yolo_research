# FCA-P0 final GT coverage audit

- Verdict: **NO-GO_ON_CLEAN_VOC**
- Pooled O2O BF16 avoidable gap: 31/240348 (0.0129%)
- Image-cluster bootstrap 95% CI: [0.0087%, 0.0176%]
- Zero-quality contribution share: 0.00%

## Frozen-checkpoint results

| checkpoint | precision | valid GT | gap | gap rate | r>=0.50 gap | retention |
|---|---:|---:|---:|---:|---:|---:|
| epoch0.pt | BF16 | 40058 | 28 | 0.0699% | 26 | 92.86% |
| epoch10.pt | BF16 | 40058 | 1 | 0.0025% | 1 | 100.00% |
| epoch20.pt | BF16 | 40058 | 0 | 0.0000% | 0 | n/a |
| epoch30.pt | BF16 | 40058 | 0 | 0.0000% | 0 | n/a |
| epoch40.pt | BF16 | 40058 | 0 | 0.0000% | 0 | n/a |
| last.pt | BF16 | 40058 | 2 | 0.0050% | 2 | 100.00% |
| last.pt | FP32 | 40058 | 2 | 0.0050% | 2 | 100.00% |

## Automatic gates

- FAIL — `bootstrap_lower_ge_0_25pct`
- FAIL — `four_checkpoint_effect`
- FAIL — `four_checkpoint_quality_retention`
- FAIL — `late_checkpoint_pass`
- PASS — `zero_quality_share_lt_25pct`
- PASS — `last_bf16_fp32_same_side`
- PASS — `geometry_identical`
- PASS — `native_coverage_rematch_feasible`

## Native stage decomposition

| checkpoint | precision | head | valid | S0 cov | S1 cov | S2 cov | S3 cov | S1→S2 lost | S1→S2 gained | added edges | off-graph final edges | S2→S3 lost | avoidable gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| epoch0.pt | BF16 | O2M | 40058 | 100.000% | 99.998% | 99.925% | 99.925% | 29 | 0 | 67 | 67 | 0 | 29 |
| epoch0.pt | BF16 | O2O | 40058 | 100.000% | 100.000% | 99.930% | 99.930% | 28 | 0 | 73 | 0 | 0 | 28 |
| epoch10.pt | BF16 | O2M | 40058 | 100.000% | 100.000% | 99.983% | 99.983% | 7 | 0 | 23 | 23 | 0 | 7 |
| epoch10.pt | BF16 | O2O | 40058 | 100.000% | 100.000% | 99.998% | 99.998% | 1 | 0 | 11 | 0 | 0 | 1 |
| epoch20.pt | BF16 | O2M | 40058 | 100.000% | 100.000% | 99.935% | 99.935% | 26 | 0 | 17 | 17 | 0 | 26 |
| epoch20.pt | BF16 | O2O | 40058 | 100.000% | 100.000% | 100.000% | 100.000% | 0 | 0 | 17 | 0 | 0 | 0 |
| epoch30.pt | BF16 | O2M | 40058 | 100.000% | 100.000% | 99.923% | 99.923% | 31 | 0 | 12 | 12 | 0 | 31 |
| epoch30.pt | BF16 | O2O | 40058 | 100.000% | 100.000% | 100.000% | 100.000% | 0 | 0 | 15 | 0 | 0 | 0 |
| epoch40.pt | BF16 | O2M | 40058 | 100.000% | 100.000% | 99.903% | 99.903% | 39 | 0 | 6 | 6 | 0 | 39 |
| epoch40.pt | BF16 | O2O | 40058 | 100.000% | 100.000% | 100.000% | 100.000% | 0 | 0 | 17 | 0 | 0 | 0 |
| last.pt | BF16 | O2M | 40058 | 100.000% | 100.000% | 99.908% | 99.908% | 37 | 0 | 13 | 13 | 0 | 37 |
| last.pt | BF16 | O2O | 40058 | 100.000% | 100.000% | 99.995% | 99.995% | 2 | 0 | 19 | 0 | 0 | 2 |
| last.pt | FP32 | O2M | 40058 | 100.000% | 100.000% | 99.903% | 99.903% | 39 | 0 | 14 | 14 | 0 | 39 |
| last.pt | FP32 | O2O | 40058 | 100.000% | 100.000% | 99.995% | 99.995% | 2 | 0 | 18 | 0 | 0 | 2 |

## O2M sidecar control

- O2M avoidable gap: 169/240348 (0.0703%).
- O2M gap is at least O2O gap; withdraw O2O-specific wording and treat this as a general TAL conflict-order issue.

## Evidence for and against continuing

- For: 4/8 preregistered gates passed.
- Against: 4 gates failed; failed gates are ['bootstrap_lower_ge_0_25pct', 'four_checkpoint_effect', 'four_checkpoint_quality_retention', 'late_checkpoint_pass'].
- The dense-tiny cell trigger is explanatory only; its quality retention uses the same fixed identity-preserving unrestricted witness edge, not a unique subgroup-optimal r>=0.50 matching.

## Quality evidence

```json
{
  "epoch0.pt:BF16:O2M:common_delta": {
    "count": 40028,
    "mean": 0.009434932460518542,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch0.pt:BF16:O2M:new_ciou_clipped": {
    "count": 29,
    "mean": 0.22347412129928326,
    "median": 0.21236950159072876,
    "p10": 0.11572728753089905,
    "p25": 0.16169050335884094,
    "p75": 0.2647653818130493,
    "p90": 0.31250687241554254
  },
  "epoch0.pt:BF16:O2M:new_iou": {
    "count": 29,
    "mean": 0.25400562128055765,
    "median": 0.24060306142448643,
    "p10": 0.14417405893239862,
    "p25": 0.19992648200039667,
    "p75": 0.31323536288167786,
    "p90": 0.3592451768446989
  },
  "epoch0.pt:BF16:O2M:new_r": {
    "count": 29,
    "mean": 0.40111599835988376,
    "median": 0.4270329149889937,
    "p10": 0.08819881436226194,
    "p25": 0.20813766188773972,
    "p75": 0.5625024365575327,
    "p90": 0.6371827129360246
  },
  "epoch0.pt:BF16:O2M:rewritten": {
    "count": 40028,
    "mean": 0.02730588587988408,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch0.pt:BF16:O2O:common_delta": {
    "count": 40030,
    "mean": 0.007716187963480154,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch0.pt:BF16:O2O:new_ciou_clipped": {
    "count": 28,
    "mean": 0.15422022648687875,
    "median": 0.1474030241370201,
    "p10": 0.09121040627360344,
    "p25": 0.1090974435210228,
    "p75": 0.19010308012366295,
    "p90": 0.23146277070045476
  },
  "epoch0.pt:BF16:O2O:new_iou": {
    "count": 28,
    "mean": 0.17383514408657552,
    "median": 0.1619844284011094,
    "p10": 0.11087473959625016,
    "p25": 0.12562952424699486,
    "p75": 0.2070748563573008,
    "p90": 0.2663601287713797
  },
  "epoch0.pt:BF16:O2O:new_r": {
    "count": 28,
    "mean": 0.4862559019066181,
    "median": 0.48994822764114365,
    "p10": 0.13025690056345698,
    "p25": 0.2954920991467201,
    "p75": 0.6506680160618561,
    "p90": 0.8297200085215113
  },
  "epoch0.pt:BF16:O2O:rewritten": {
    "count": 40030,
    "mean": 0.030402198351236573,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch10.pt:BF16:O2M:common_delta": {
    "count": 40051,
    "mean": 0.0026127923378993427,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch10.pt:BF16:O2M:new_ciou_clipped": {
    "count": 7,
    "mean": 0.6228508523532322,
    "median": 0.6708568334579468,
    "p10": 0.45049726963043213,
    "p25": 0.5735785663127899,
    "p75": 0.7015857398509979,
    "p90": 0.7394789099693299
  },
  "epoch10.pt:BF16:O2M:new_iou": {
    "count": 7,
    "mean": 0.6300070701300522,
    "median": 0.6764349510954132,
    "p10": 0.46522297941159724,
    "p25": 0.5841595193413964,
    "p75": 0.7063776955754308,
    "p90": 0.7440728088146674
  },
  "epoch10.pt:BF16:O2M:new_r": {
    "count": 7,
    "mean": 0.5717602965838067,
    "median": 0.6057941876824283,
    "p10": 0.40004223353956064,
    "p25": 0.45381188289380187,
    "p75": 0.6600574808514373,
    "p90": 0.7115643044152145
  },
  "epoch10.pt:BF16:O2M:rewritten": {
    "count": 40051,
    "mean": 0.007290704351951262,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch10.pt:BF16:O2O:common_delta": {
    "count": 40057,
    "mean": 0.004188280799208844,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch10.pt:BF16:O2O:new_ciou_clipped": {
    "count": 1,
    "mean": 0.774350643157959,
    "median": 0.774350643157959,
    "p10": 0.774350643157959,
    "p25": 0.774350643157959,
    "p75": 0.774350643157959,
    "p90": 0.774350643157959
  },
  "epoch10.pt:BF16:O2O:new_iou": {
    "count": 1,
    "mean": 0.7765918842567446,
    "median": 0.7765918842567446,
    "p10": 0.7765918842567446,
    "p25": 0.7765918842567446,
    "p75": 0.7765918842567446,
    "p90": 0.7765918842567446
  },
  "epoch10.pt:BF16:O2O:new_r": {
    "count": 1,
    "mean": 0.8852618639094209,
    "median": 0.8852618639094209,
    "p10": 0.8852618639094209,
    "p25": 0.8852618639094209,
    "p75": 0.8852618639094209,
    "p90": 0.8852618639094209
  },
  "epoch10.pt:BF16:O2O:rewritten": {
    "count": 40057,
    "mean": 0.01305639463764136,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch20.pt:BF16:O2M:common_delta": {
    "count": 40032,
    "mean": 0.0020340176602693352,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch20.pt:BF16:O2M:new_ciou_clipped": {
    "count": 26,
    "mean": 0.6581503872114879,
    "median": 0.6966697871685028,
    "p10": 0.4895416498184204,
    "p25": 0.6168006062507629,
    "p75": 0.7465526163578033,
    "p90": 0.8028874695301056
  },
  "epoch20.pt:BF16:O2M:new_iou": {
    "count": 26,
    "mean": 0.6664827903018923,
    "median": 0.7064409053232485,
    "p10": 0.5005251637759769,
    "p25": 0.6279434794788273,
    "p75": 0.7509977260627856,
    "p90": 0.8044801857805752
  },
  "epoch20.pt:BF16:O2M:new_r": {
    "count": 26,
    "mean": 0.6980561753947276,
    "median": 0.7381382106637538,
    "p10": 0.38805765974665973,
    "p25": 0.5319343822039817,
    "p75": 0.8654837897851229,
    "p90": 0.9616285610719357
  },
  "epoch20.pt:BF16:O2M:rewritten": {
    "count": 40032,
    "mean": 0.005820343725019984,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch20.pt:BF16:O2O:common_delta": {
    "count": 40058,
    "mean": 0.003729272460227212,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch20.pt:BF16:O2O:rewritten": {
    "count": 40058,
    "mean": 0.011283638723850417,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch30.pt:BF16:O2M:common_delta": {
    "count": 40027,
    "mean": 0.0015833934939262241,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch30.pt:BF16:O2M:new_ciou_clipped": {
    "count": 31,
    "mean": 0.6666970339513594,
    "median": 0.6929734945297241,
    "p10": 0.489808052778244,
    "p25": 0.6164828240871429,
    "p75": 0.7366126477718353,
    "p90": 0.7881133556365967
  },
  "epoch30.pt:BF16:O2M:new_iou": {
    "count": 31,
    "mean": 0.6743822933285546,
    "median": 0.6958208414952355,
    "p10": 0.49590433077397095,
    "p25": 0.6234852738931373,
    "p75": 0.7420243032224356,
    "p90": 0.7892521893127492
  },
  "epoch30.pt:BF16:O2M:new_r": {
    "count": 31,
    "mean": 0.7550773042253437,
    "median": 0.8034599013440825,
    "p10": 0.42975275556533754,
    "p25": 0.5814594756910846,
    "p75": 0.9168721989314181,
    "p90": 0.9907548629441766
  },
  "epoch30.pt:BF16:O2M:rewritten": {
    "count": 40027,
    "mean": 0.0048467284582906535,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch30.pt:BF16:O2O:common_delta": {
    "count": 40058,
    "mean": 0.0029922644783877154,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch30.pt:BF16:O2O:rewritten": {
    "count": 40058,
    "mean": 0.009486244944829997,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch40.pt:BF16:O2M:common_delta": {
    "count": 40019,
    "mean": 0.001491675408352209,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch40.pt:BF16:O2M:new_ciou_clipped": {
    "count": 39,
    "mean": 0.6821526288986206,
    "median": 0.6973421573638916,
    "p10": 0.5763276100158692,
    "p25": 0.6214976608753204,
    "p75": 0.7394965887069702,
    "p90": 0.7919121384620668
  },
  "epoch40.pt:BF16:O2M:new_iou": {
    "count": 39,
    "mean": 0.6889004761173118,
    "median": 0.7041865180922933,
    "p10": 0.5902797401634782,
    "p25": 0.6313497236397668,
    "p75": 0.745466847913016,
    "p90": 0.7942635692502948
  },
  "epoch40.pt:BF16:O2M:new_r": {
    "count": 39,
    "mean": 0.7592185147064584,
    "median": 0.7875165193349144,
    "p10": 0.43747052287756183,
    "p25": 0.6365119801787416,
    "p75": 0.9344924970794215,
    "p90": 0.997600318859243
  },
  "epoch40.pt:BF16:O2M:rewritten": {
    "count": 40019,
    "mean": 0.004148029685899198,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch40.pt:BF16:O2O:common_delta": {
    "count": 40058,
    "mean": 0.002418485571483179,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "epoch40.pt:BF16:O2O:rewritten": {
    "count": 40058,
    "mean": 0.007089719906136102,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "last.pt:BF16:O2M:common_delta": {
    "count": 40021,
    "mean": 0.001397420526187089,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "last.pt:BF16:O2M:new_ciou_clipped": {
    "count": 37,
    "mean": 0.6347777376303801,
    "median": 0.6630754470825195,
    "p10": 0.4656896471977234,
    "p25": 0.5851523280143738,
    "p75": 0.7164892554283142,
    "p90": 0.7607827425003051
  },
  "last.pt:BF16:O2M:new_iou": {
    "count": 37,
    "mean": 0.6431438256884298,
    "median": 0.6709703859257082,
    "p10": 0.47308697039971964,
    "p25": 0.6009040184698193,
    "p75": 0.7222371666512575,
    "p90": 0.7653379440720927
  },
  "last.pt:BF16:O2M:new_r": {
    "count": 37,
    "mean": 0.747205272759505,
    "median": 0.8260990156413005,
    "p10": 0.5244511435185971,
    "p25": 0.6121798689323346,
    "p75": 0.8882726927980165,
    "p90": 0.9507028748075765
  },
  "last.pt:BF16:O2M:rewritten": {
    "count": 40021,
    "mean": 0.0036230978736163516,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "last.pt:BF16:O2O:common_delta": {
    "count": 40056,
    "mean": 0.002125871878719808,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "last.pt:BF16:O2O:new_ciou_clipped": {
    "count": 2,
    "mean": 0.29495829343795776,
    "median": 0.29495829343795776,
    "p10": 0.15021272897720336,
    "p25": 0.20449231564998627,
    "p75": 0.38542427122592926,
    "p90": 0.43970385789871214
  },
  "last.pt:BF16:O2O:new_iou": {
    "count": 2,
    "mean": 0.31674246177184645,
    "median": 0.31674246177184645,
    "p10": 0.173455809222749,
    "p25": 0.22718830392866055,
    "p75": 0.4062966196150324,
    "p90": 0.4600291143209439
  },
  "last.pt:BF16:O2O:new_r": {
    "count": 2,
    "mean": 0.08740011529208491,
    "median": 0.08740011529208491,
    "p10": 0.017588170256291592,
    "p25": 0.04376764964471408,
    "p75": 0.13103258093945575,
    "p90": 0.1572120603278782
  },
  "last.pt:BF16:O2O:rewritten": {
    "count": 40056,
    "mean": 0.006715598162572399,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "last.pt:FP32:O2M:common_delta": {
    "count": 40019,
    "mean": 0.0014471090650020084,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "last.pt:FP32:O2M:new_ciou_clipped": {
    "count": 39,
    "mean": 0.6350294481485318,
    "median": 0.6670652031898499,
    "p10": 0.4463458895683289,
    "p25": 0.5873643755912781,
    "p75": 0.7164603173732758,
    "p90": 0.7570925474166871
  },
  "last.pt:FP32:O2M:new_iou": {
    "count": 39,
    "mean": 0.6434845807868876,
    "median": 0.6716930719218317,
    "p10": 0.4596143104391522,
    "p25": 0.6043800149875107,
    "p75": 0.7218912134157419,
    "p90": 0.762015914757414
  },
  "last.pt:FP32:O2M:new_r": {
    "count": 39,
    "mean": 0.7510116415720667,
    "median": 0.8129035069066133,
    "p10": 0.514945659873438,
    "p25": 0.6335296559880098,
    "p75": 0.886950312803396,
    "p90": 0.9259759885947745
  },
  "last.pt:FP32:O2M:rewritten": {
    "count": 40019,
    "mean": 0.0036732552037782055,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "last.pt:FP32:O2O:common_delta": {
    "count": 40056,
    "mean": 0.002100375368139223,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  },
  "last.pt:FP32:O2O:new_ciou_clipped": {
    "count": 2,
    "mean": 0.30691055208444595,
    "median": 0.30691055208444595,
    "p10": 0.15409082919359207,
    "p25": 0.21139822527766228,
    "p75": 0.40242287889122963,
    "p90": 0.45973027497529984
  },
  "last.pt:FP32:O2O:new_iou": {
    "count": 2,
    "mean": 0.32927554487467964,
    "median": 0.32927554487467964,
    "p10": 0.17835884322854573,
    "p25": 0.23495260634584594,
    "p75": 0.42359848340351336,
    "p90": 0.48019224652081355
  },
  "last.pt:FP32:O2O:new_r": {
    "count": 2,
    "mean": 0.10745626146114556,
    "median": 0.10745626146114556,
    "p10": 0.021615686293062146,
    "p25": 0.053805901981093425,
    "p75": 0.1611066209411977,
    "p90": 0.19329683662922897
  },
  "last.pt:FP32:O2O:rewritten": {
    "count": 40056,
    "mean": 0.006241262232873976,
    "median": 0.0,
    "p10": 0.0,
    "p25": 0.0,
    "p75": 0.0,
    "p90": 0.0
  }
}
```

## Interpretation boundary

This was a zero-training train-split assignment audit. It did not read VOC2007 test AP, update weights, or establish that any proposed matcher improves detection accuracy. A GO verdict authorizes only the pre-registered offline Phase 1 comparisons.
Native S2 is reported as a non-monotone rewiring stage: the frozen implementation may introduce an edge outside E_topk via an all-GT overlap argmax. The main matching graph remains strictly E_topk; native off-graph pairs are never inserted into it.

## Runtime scans

- epoch0.pt BF16: 16551 images, 40058 GT, 63.0s, geometry `05806c56188ad7e29e589d68be80362a1e8ef4553dac344f9441215cfc9f206e`
- epoch10.pt BF16: 16551 images, 40058 GT, 62.4s, geometry `05806c56188ad7e29e589d68be80362a1e8ef4553dac344f9441215cfc9f206e`
- epoch20.pt BF16: 16551 images, 40058 GT, 62.6s, geometry `05806c56188ad7e29e589d68be80362a1e8ef4553dac344f9441215cfc9f206e`
- epoch30.pt BF16: 16551 images, 40058 GT, 62.6s, geometry `05806c56188ad7e29e589d68be80362a1e8ef4553dac344f9441215cfc9f206e`
- epoch40.pt BF16: 16551 images, 40058 GT, 62.8s, geometry `05806c56188ad7e29e589d68be80362a1e8ef4553dac344f9441215cfc9f206e`
- last.pt BF16: 16551 images, 40058 GT, 63.1s, geometry `05806c56188ad7e29e589d68be80362a1e8ef4553dac344f9441215cfc9f206e`
- last.pt FP32: 16551 images, 40058 GT, 68.0s, geometry `05806c56188ad7e29e589d68be80362a1e8ef4553dac344f9441215cfc9f206e`
