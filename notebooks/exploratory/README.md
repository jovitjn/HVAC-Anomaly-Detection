# Original exploration

`RTU_supervised_original.ipynb` is the Colab notebook I used while testing the
first versions of the model. I kept it unchanged, including the saved outputs,
because it records how the idea developed across several Conv1D experiments.

The notebook is exploratory rather than the reference implementation. In
particular, some cells use `-999` for missing values, fit preprocessing before
the temporal split, repeat the reconstruction decoder for every room, or use a
fixed threshold without a separate calibration set. The cleaned pipeline in
`src/hvac_anomaly/` addresses those issues and produces the reported results.

