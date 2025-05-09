# SpecSTG: A Spectral Diffusion Framework for Fast Probabilistic Spatio-Temporal Graph Forecasting [ICLR2025 DeLTa Workshop]

## Description
This is the source code for “SpecSTG: A Spectral Diffusion Framework for Fast Probabilistic Spatio-Temporal Graph Forecasting”. SpecSTG is a diffusion model for spatio-temporal graph forecasting that generates the graph Fourier representation of time series.

Please "star" us if you find the code helpful ✧◝(⁰▿⁰)◜✧

## Model Architecture
![Graph Figure](https://github.com/user-attachments/assets/a61b9cd4-aed8-408d-9c42-0b51fd10c4e4)

## Requirements
```bash 
torch 2.0.1
torch_geometric 2.3.1
pytorchts == 0.6.0
gluonts == 0.10.0
```

## Datasets
In the paper, we evaluate SpecSTG with traffic flow and vehicle speed forecasting tasks. Our experiments include two traffic datasets, PEMS04 and PEMS08. Both datasets can be retrieved from [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data). 

## Training
```bash
python SpecSTG_exe.py -- load_model = False
```

## Evaluation
```bash
python SpecSTG_exe.py -- load_model = True
```

## Recommended Reading

✧ Diffusion models for time-series applications: a survey. Frontiers of Information Technology & Electronic Engineering, 25(1), 19-41. Available at [[Springer]](https://link.springer.com/article/10.1631/FITEE.2300310).


✧ Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting. In International Conference on Machine Learning (pp. 8857-8868).


✧ Graph convolution recurrent denoising diffusion model for multivariate probabilistic temporal forecasting. In International Conference on Advanced Data Mining and Applications (pp. 661-676). 


