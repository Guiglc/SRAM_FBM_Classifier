你现在可以这样跑：

python SRAM_FBM_Classifier.py "D:\00 My Work\09 Programming\log_parser\T5833_parser\FBM\AS00706_55HV_277_normal_failbit.csv" --outdir result

如果你的 macro 是 512 × 512：

python sram_pattern_classifier.py failbit.csv --macro-cols 512 --macro-rows 512 --outdir result

如果你后面发现 `PERIODIC` 误判偏多，可以提高：

--periodic-min-points 6
--periodic-ratio 0.8
--periodic-min-lines 2

这版程序先按你现在的分类表实现了主流程：

MACRO_BLOCK
PERIODIC
SWR / SBC
PWR / PBC
MWR / MBC
CROSS
BLOCK
QUADRA_BIT
RANDOM_CLUSTER
SB / DBR / DBC / L-shape
RANDOM_DENSITY

![1778067698989](image/note/1778067698989.png)![1778067728560](image/note/1778067728560.png)![1778068117500](image/note/1778068117500.png)

![1778068129443](image/note/1778068129443.png)


![1778068285631](image/note/1778068285631.png)
