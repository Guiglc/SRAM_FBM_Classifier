你现在可以这样跑：

python sram_pattern_classifier.py failbit.csv --outdir result

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
