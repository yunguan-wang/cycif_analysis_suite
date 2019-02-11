# cycif_analysis_suite
A collection of tools for analysis of segmented single-cell cycif data. Functionalities includes ellimination of bad/lost cells, differential analysis, dimention reduction and clustering.
## Installation
```
git clone https://github.com/labsyspharm/cycif_analysis_suite.git
cd cycif_analysis_suite
pip install -e .
```
## Example
To test run cycif_analysis_suite, simple run the example.py in the cloned repository folder. The examplar data is a small subset of a real cycif dataset.
```
python example.py
```
After a few minutes, you should see the result tables and figures in the `example_output` folder.

## Tutorial
Modules in this suite can be divided into several modules including QC, contrast-based differential analysis and unsurpervised analyses. 

### QC (elliminate bad/lost cells)
During a washing cycle in cycif, cells can be pushed away or even washed off. If a segmention masked is placed on such cells, close to background signals will be recorded for channels of the subsequent cycles, although these cells should in fact be treated as missing data. Since this procedure is repeated several times in a experiment, the cell loss due to this cannot be ignored. Thus the first procedure of data analysis should be getting rid of these cells. 
Currently, a simple thresholding based method is used to detect moved/washed-away cells. Thresholds can be applied on intensity difference between adjacent cycles of the DAPI channel or on background intensity of DAPI channel. The exact threshold value is determined from data.
Determining threshold with cycle difference.
```
from cycifsuite.detect_lost_cells import ROC_lostcells, get_lost_cells, plot_lost_cell_stacked_area
import pandas as pd
raw_qc_expr_data = pd.read_csv('example_input/sample_raw_data_for_QC.csv', index_col=0)
_,_,t_c = ROC_lostcells(raw_qc_expr_data, cutoff_min=0, cutoff_max=15,
                        steps=50, filtering_method='cycle_diff', figname='example_output/Cycle difference thresholding.png')
```
The threshold is determined to be the elbow point of the curve, as shown as the intersection of red dashed line and the curve.

<img src="https://github.com/labsyspharm/cycif_analysis_suite/blob/MCF10A/example_output/Cycle difference thresholding.png" width="400">

Then, lost cells with the derived threshold can be extracted and visualized
```
df_lc_cv, _ = get_lost_cells(raw_qc_expr_data, t_c,8,'cycle_diff')
plot_lost_cell_stacked_area(df_lc_cv, figname='example_output/accumulated cell loss by cycle.png')
```

<img src="https://github.com/labsyspharm/cycif_analysis_suite/blob/MCF10A/example_output/accumulated cell loss by cycle.png" width="400">

### Contrast based differential analysis
This module is design for analysis comparing test conditions vs a reference condition (control), and thus is best suited for plate-based cycif data.
Read expression data and metadata. Metadata should have columns specifying treatment conditions and replicate.
```
expr_data = pd.read_csv('example_input/sample_exprdata.csv', index_col=0)
metadata = pd.read_csv('example_input/sample_metadata.csv', index_col=0)
```

Specify control condition and output file prefix.
```
control_name = 'ctrl'
output_prefix = 'T0_ctrl as control'
```

Load differential analysis module. Column names for treatment conditions and replicate need to be passed.
```
pwa = per_well_analysis(expr_data, metadata, batch_col='replicate',
                        drug_col='ligand', dose_col='dose', time_col='time', control_name=control_name,
                        output_path='example_output', output_prefix=output_prefix)
                        
```

Make contrasts, which are pairs of test and control conditions.
```
pwa.make_contrast(batch_wise=True)
```

Overall distance to control condition can be evaluated and plotted.
```
pwa.contrast_based_distance()
```
![alt text](https://github.com/labsyspharm/cycif_analysis_suite/blob/MCF10A/example_output/Drug_control%20distance%20over%20time%20per%20ligand.png)

Perform T-test and kolmogorov-smirnov test for each pair comparing test with control condition.
```
pwa.contrast_based_differential_analysis()
```
Out put table with log fold change of each channel comparing test vs control.

Channel | logFC |	T_pValue | KS_pValue | adj_T_pVal | adj_KS_pVal	| ligand	| dose	| time	| Batch	Control_metadata
-------------| ---- | ---- | ---- | ---- | ---- | ---- |----|----|--------
Cytoplasm_Catenin (Beta)|-0.13|0.65|0.97|0.85|0.97|BMP2|20|1|ctrl_0_0
Cytoplasm_Cyclin D1|-0.36|0.37|0.11|0.67|0.23|BMP2|20|1|ctrl_0_0

Make a dotted heatmap summerizing logFC of all channels over all conditions.
```
pwa.make_fc_heamtap_matrix(['ligand', 'time', 'Batch'], output_fname=output_prefix)
pwa.make_dotted_heatmap()
```
![alt text](https://github.com/labsyspharm/cycif_analysis_suite/blob/MCF10A/example_output/T0_ctrl%20as%20controllogFC%20dotted%20heatmap.png)

To evaluate the distribution of fold change across each replicate, a swarm plot can be used.
```
pwa.make_swarm_plot()
```
![alt text](https://github.com/labsyspharm/cycif_analysis_suite/blob/MCF10A/example_output/T0_ctrl%20as%20control%20swarm%20logFC%20plot%20by%20time%20.png)
