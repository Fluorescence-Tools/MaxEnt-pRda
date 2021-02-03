# MaxEnt-pRda
Re-weighting of conformational ensembles directed by the inter-dye distance distributions derived from FRET experiments.

## Usage

```
./optimize_lif_mem.py 50000 "Lif_data/MD_Neha_pre2021/results" 0.2
```
First argument (`50000`) is the maximum number of optimization iterations.
Second argument is the path to the directory to save the results (`"Lif_data/MD_Neha_pre2021/results"`).
Third argument (`0.2`) is `θ`. θ is the tuning parameter that allows to chose the relative weight of `χ2` and entropy (`S`). For `θ = 0` the entropy is ignored and the algorithm minimizes the `χ2`, for `θ = +∞`, `χ2` is ignored and no reweighing can happen since the original weights are fully preserved.
This example runs MEM optimization with 30000 steps and `θ = 0.2`. The results are saved in the `Lif_data/MD_Neha_pre2021/results` directory.
The results include the [optimized cluster weights](Lif_data/MD_Neha_pre2021/results/weights_final.dat), [optimized inter-dye distance distributions](Lif_data/MD_Neha_pre2021/results/pRda_model_137_215.dat), and corresponding plots for each FRET pair.

![inter-dye distance distributions before and after MEM optimization](Lif_data/MD_Neha_pre2021/results/268_296_49800.png)
