# SRAM FBM Classifier

## Usage

Edit `INPUT_PATH` in `SRAM_FBM_Classifier.py`:

```python
INPUT_PATH = r"E:\path\to\failbit.csv"
```

Then run:

```bash
python SRAM_FBM_Classifier.py
```

Results are written to a `result` folder next to the input CSV:

- `pattern_summary.csv`
- `failbit_labeled.csv`

<<<<<<< Updated upstream
![1778067698989](image/note/1778067698989.png)![1778067728560](image/note/1778067728560.png)
=======
## Current Rules

The classifier uses the rule table in `class.xlsx`.
>>>>>>> Stashed changes

Active pattern codes:

<<<<<<< Updated upstream
![1778131410001](image/note/1778131410001.png)![1778131715463](image/note/1778131715463.png)![1778131832962](image/note/1778131832962.png)![1778131878373](image/note/1778131878373.png)
=======
- `MBLK`: macro block
- `SWR`, `SBC`: full row / full column
- `PWR`, `PBC`: partial row / partial column, continuous length `> 20`
- `MWR`, `MBC`: adjacent multi-row / multi-column
- `CRS`: row-type pattern intersects column-type pattern
- `BLK`: block, default size threshold `>= 25`
- `QB`: 2x2 quadra bit
- `RCLU`: random cluster, connected component size `>= 3`
- `DWR`, `DWC`: dotted row / dotted column
- `SB`, `DBR`, `DBC`, `TB`: small local patterns
- `RDEN`: random density override

Removed pattern codes:

- `PDIC`
- `TBR`
- `TBC`

`L-shape` is now named `TB`.

## RDEN Override

After normal classification, the classifier counts small random-like modes in each macro:

- `SB`
- `DBR`
- `DBC`
- `QB`
- `TB`
- `RCLU`

If the count is greater than `200`, those modes are overwritten as `RDEN` in both summary and labeled outputs. The original type is preserved in `original_pattern_code` and `original_pattern_name` in `pattern_summary.csv`.
>>>>>>> Stashed changes
