# SpecSTG: A Spectral Diffusion Framework for Fast Probabilistic Spatio-Temporal Graph Forecasting

## Description
This is the source code for “SpecSTG: A Spectral Diffusion Framework for Fast Probabilistic Spatio-Temporal Graph Forecasting”, available at [arXiv](https://arxiv.org/abs/2401.08119). SpecSTG is a diffusion model for spatio-temporal graph forecasting that generates the graph Fourier representation of time series. 

## Model Architecture
![Graph Figure](https://github.com/user-attachments/assets/a61b9cd4-aed8-408d-9c42-0b51fd10c4e4)

## Requirements
```bash 
torch 2.3.0
torch_geometric 2.3.1
pytorchts 0.6.0
gluonts 0.10.0
```



## Datasets
In the paper, we evaluate SpecSTG with traffic flow and vehicle speed forecasting tasks. Our experiments include two traffic datasets, PEMS04 and PEMS08. Both datasets can be retrieved from [ASTGNN](https://github.com/guoshnBJTU/ASTGNN/tree/main/data). 



