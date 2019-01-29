# cycif_analysis_suite
A collection of tools for analysis of segmented single-cell cycif data. Functionalities includes ellimination of bad/lost cells, differential analysis, dimention reduction and clustering.
## Installation
```
git clone https://github.com/yunguan-wang/cycif_analysis_suite.git
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
TODO

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
![alt text](https://github.com/yunguan-wang/cycif_analysis_suite/blob/MCF10A/example_output/Drug_control%20distance%20over%20time%20per%20ligand.png)

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
![alt text](https://github.com/yunguan-wang/cycif_analysis_suite/blob/MCF10A/example_output/T0_ctrl%20as%20controllogFC%20dotted%20heatmap.png)
