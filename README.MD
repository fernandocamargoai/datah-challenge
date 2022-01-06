# Data H Challenge

This is the source code used to train models for Demand Forecasting using GluonTS and PyTorch.

The data for the competition can be found at: https://www.kaggle.com/c/datah-m1-challange

To run the code, first download the data as follows:

```bash
cd assets
kaggle competitions download -c datah-m1-challange
unzip datah-m1-challange.zip
```

Then, to run an experiment, execute:

```bash
PYTHONPATH=. luigi --module datah_challenge.task.training DeepARTraining --max-epochs 30
```
