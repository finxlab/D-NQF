# Decomposition-Based Neural Quantile Function for Multi-Step Probabilistic Wind Speed Forecasting

<img width="9828" height="2466" alt="fig2_D-NQF" src="https://github.com/user-attachments/assets/2db46474-5e5c-46e0-bd45-72d12690b4aa" />




## Getting Started

```
# 1. Create Conda Environment
conda create -n DNQF python=3.10
conda activate DNQF

# 2. Install PyTorch
pip install torch torchvision torchaudio  # CUDA Version

# 3. Install Dependencies
pip install -r requirements.txt
```

## Data
NOAA Wind Speed Data (2022-2024) for Barrow (BRW), Mauna Loa (MLO), and South Pole (SPO). 
Download wind speed data browser (https://gml.noaa.gov/aftp/data/meteorology/in-situ/).


## Model Training
We provide scripts to train the model across the full range of hyperparameters used in our study.
The training module is designed to iterate through the complete hyperparameter space to replicate the results reported in the paper.

'''
sh scripts.sh
'''

## Model Test
'''
python test.py --data brw --pred_len 24
'''

​
## Evaluation & Analysis
The evaluation process is handled in two ways:

Quick Test: A simplified testing script is provided for a fast sanity check of the model performance.

Full Benchmark Analysis: For our actual research findings, we followed a more rigorous process:

  Test results from all benchmarks are exported and saved as numpy (.npy) files.

  Each metric is then calculated independently from these saved raw outputs to ensure precision and consistency across different evaluation environments.


📂 Repository Structure
data/: Scripts for data downloading and preprocessing.
train.py: Main entry point for training (supports full hyperparameter sweeps).
test.py: Simplified evaluation script.
