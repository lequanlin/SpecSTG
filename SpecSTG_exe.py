import numpy as np
import pandas as pd
import torch
import random
import time
from data.dataloader_stg import DataLoader_STG
from data.spectral_transform import SpecTransform
from algorithm.SpecSTG_estimator import SpecSTGEstimator
from utils.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from utils.evaluation import masked_mse_np, masked_mae_np, masked_mape_np, calc_quantile_CRPS
import argparse

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Entries of this code
parser = argparse.ArgumentParser(description='Entry point of the code')

# Settings
parser.add_argument("--model", type = str, default = 'SpecSTG', help = 'Model name')
parser.add_argument("--load_model", type = bool, default = True, help = 'True for test mode: load saved model for sampling')
parser.add_argument("--seed", type = int, default = 20240108, help = 'Random seed')

# Dataset
parser.add_argument("--dataset", type = str, default = 'PEMS04F', help = 'Dataset available: PEMS04F, PEMS04S, PEMS08F, PEMS08S')
parser.add_argument("--test_pct", type = float, default = 0.2, help = 'Test set ratio')
parser.add_argument("--val_pct", type = float, default = 0.2, help = 'Validation set ratio')
parser.add_argument("--normalization", type = bool, default = True, help = 'Whether to apply Z-score normalization')

# Task
parser.add_argument("--history_window", type = int, default = 12, help = 'Length of past time series')
parser.add_argument("--prediction_window", type = int, default = 12, help = 'Length of future time series')


# Model
parser.add_argument("--diffusion_steps", type = int, default = 50, help = 'Number of diffusion steps')
parser.add_argument("--beta_end", type =  float, default = 0.2, help = 'Maximum beta value')
parser.add_argument("--beta_schedule", type = str, default = 'quad', help = 'Beta schedule')
parser.add_argument("--K", type = int, default = 2, help = 'SpecConv polynomial level')
parser.add_argument("--num_cells", type = int, default = 96, help = 'RNN hidden size')


# Training
parser.add_argument("--max_epochs", type = int, default = 300, help = 'Maximum  number of epochs')
parser.add_argument("--lr", type = float, default =5e-4, help = 'Start learning rate')
parser.add_argument("--max_lr", type = float, default = 1e-2, help = 'End learning rate')
parser.add_argument("--weight_decay", type = float, default = 1e-6, help = 'Weight decay term')
parser.add_argument("--dropout", type = float, default = 0.1, help = 'Dropout probability')
parser.add_argument("--num_batches_per_epoch", type = int, default = 200, help = 'Number of batches per epoch')
parser.add_argument("--num_batches_per_epoch_val", type = int, default = 50, help = 'Number of batches per epoch for validation')
parser.add_argument("--batch_size", type = int, default = 8, help = 'Batch size')
parser.add_argument("--patience", type = int, default = 300, help = 'Early stopping patience')

# Testing
parser.add_argument("--num_samples", type = int, default = 200, help = 'Number of samples')

args = parser.parse_args()

# Save model and rng paths
path_model = "./model_storage/SpecSTG_{}_step{}_beta{}_cell{}_lr{}_wd{}_dp{}.pt".format(args.dataset,
                                                                  args.diffusion_steps, args.beta_end, args.num_cells,
                                                                                   args.lr, args.weight_decay, args.dropout)
path_rng = "./model_storage/SpecSTG_{}_step{}_beta{}_cell{}_lr{}_wd{}_dp{}_rng.pt".format(args.dataset,
                                                                  args.diffusion_steps, args.beta_end, args.num_cells, args.lr,
                                                                                     args.weight_decay, args.dropout)

# Random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(args.seed)

# Load data & Get Fourier representation
dataset = DataLoader_STG(normalization = args.normalization)
dataset(dataset_name = args.dataset)
X, X_norm, edge_weight, edge_index, input_size, mean, std = dataset.X, dataset.X_norm, \
    dataset.edge_weight, dataset.edge_index, dataset.input_size, dataset.mean, dataset.std

transform = SpecTransform(dataset = args.dataset)
if args.normalization:
    transform(X=X_norm, edge_weight=edge_weight, edge_index=edge_index, num_nodes=X.shape[0])
else:
    transform(X=X, edge_weight=edge_weight, edge_index=edge_index, num_nodes=X.shape[0])

X_f, reconstruct_matrix, Lambda = transform.X_f, transform.reconstruct_matrix, transform.Lambda
X_f = torch.DoubleTensor(X_f)
edge_index= torch.LongTensor(edge_index).to(device)
edge_weight = torch.FloatTensor(edge_weight).to(device)

# We arrange data in a list of dictionary, each list item is a time series
period = period = pd.Period('2018-01-01', freq='5min')
time_length = X_f.shape[1]
num_nodes = X_f.shape[0]
test_length = round(args.test_pct * time_length)
val_length = round(args.val_pct * time_length)
train_length = time_length-val_length-test_length

dataset_train = [{'target': X_f[:,:-(val_length+test_length)],  'start': period, 'feat_static_cat': np.array([0]),
                  'edge_index': edge_index, 'edge_weight': edge_weight, 'Lambda': Lambda}]
dataset_val = [{'target': X_f[:,train_length:train_length+val_length], 'start': period + train_length,
                'feat_static_cat': np.array([0]), 'edge_index': edge_index, 'edge_weight': edge_weight, 'Lambda': Lambda}]

# Model Training
myTrainer = Trainer(device=device,
                    epochs= args.max_epochs,
                    learning_rate= args.lr,
                    maximum_learning_rate = args.max_lr,
                    weight_decay = args.weight_decay,
                    num_batches_per_epoch= args.num_batches_per_epoch,
                    num_batches_per_epoch_val= args.num_batches_per_epoch_val,
                    batch_size= args.batch_size,
                    path = path_model,
                    path_rng = path_rng,
                    load_model= args.load_model,
                    patience = args.patience,
                    seed = args.seed,
                   )

estimator = SpecSTGEstimator(
    num_nodes= num_nodes,
    num_instances= train_length,
    prediction_length= args.prediction_window,
    context_length=args.history_window,
    num_cells = args.num_cells,
    dropout_rate = args.dropout,
    num_parallel_samples = args.num_samples,
    freq='5min',
    diff_steps= args.diffusion_steps,
    beta_end= args.beta_end,
    beta_schedule= args.beta_schedule,
    K = args.K,
    trainer= myTrainer
)

predictor = estimator.train(training_data = dataset_train, validation_data = dataset_val, shuffle_buffer_length = train_length,
                            num_workers=0, prefetch_factor = None)



# Model testing
test_window = test_length - args.prediction_window - args.history_window
prediction_end = np.arange(0, test_length - args.prediction_window - args.history_window)

### Enable for smaller test window if needed###
# test_window = 100
# prediction_end = np.rint(np.linspace(0, test_length - args.prediction_window - args.history_window, num=test_window)).astype(int)

test_num = 0
MSE = np.zeros((test_window, 12))
MAE_med = np.zeros((test_window, 12))
MAE_mean = np.zeros((test_window, 12))
MAPE_med = np.zeros((test_window, 12))
MAPE_mean = np.zeros((test_window, 12))
CRPS = np.zeros(test_window)
CRPS_3 = np.zeros(test_window)
CRPS_6 = np.zeros(test_window)
CRPS_12 = np.zeros(test_window)
denom = 0
denom_3 = 0
denom_6 = 0
denom_12 = 0

# count time
t_start = time.time()

for k in prediction_end:

    if k == 0:
        mytarget = X_f
        true_target = X[:, - args.prediction_window:]
    else:
        mytarget = X_f[:, :-k]
        true_target = X[:, -k - args.prediction_window: -k]

    dataset_test = [{
        'target': mytarget,
        'start': period, 'feat_static_cat': np.array([0]),
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'Lambda': Lambda
    }]

    forecast, ts = make_evaluation_predictions(
        dataset=dataset_test,
        predictor=predictor,
        num_samples=args.num_samples
    )
    forecast_list = list(forecast)

    samples = forecast_list[0].samples
    samples = np.tensordot(reconstruct_matrix, samples.T,  axes=((1),(0)))

    predict_target_mean = forecast_list[0].mean
    predict_target_mean = reconstruct_matrix @ predict_target_mean.T
    predict_target_med = forecast_list[0].quantile(0.5)
    predict_target_med = reconstruct_matrix @ predict_target_med.T

    if args.normalization:
        samples = samples*std + mean
        predict_target_mean = predict_target_mean * std + mean
        predict_target_med = predict_target_med * std + mean

    samples[samples < 0] = 0
    predict_target_mean[predict_target_mean < 0] = 0
    predict_target_med[predict_target_med < 0] = 0

    samples = np.expand_dims(np.transpose(samples, (2, 1, 0)), 0)

    metric_index = 0
    for i in range(0, 12):
        pred_mean = predict_target_mean[:, i]
        pred_med = predict_target_med[:, i]
        true = true_target[:, i]

        mse = masked_mse_np(pred_mean, true, 0.0)
        MSE[test_num, metric_index] = mse

        mae_med = masked_mae_np(pred_med, true, 0.0)
        MAE_med[test_num, metric_index] = mae_med
        mae_mean = masked_mae_np(pred_mean, true, 0.0)
        MAE_mean[test_num, metric_index] = mae_mean

        mape_med = masked_mape_np(pred_med, true, 0.0)
        MAPE_med[test_num, metric_index] = mape_med
        mape_mean = masked_mape_np(pred_mean, true, 0.0)
        MAPE_mean[test_num, metric_index] = mape_mean

        metric_index = metric_index + 1

    true_target_crps = np.expand_dims(true_target.T, 0)
    eval_points = np.ones_like(true_target_crps)
    CRPS[test_num], denom_new = calc_quantile_CRPS(torch.from_numpy(true_target_crps), torch.from_numpy(samples),
                                                   torch.from_numpy(eval_points))
    denom += denom_new

    true_target_crps_3 = np.expand_dims(true_target[:, 2].T, 0)
    eval_points_3 = np.ones_like(true_target_crps_3)
    CRPS_3[test_num], denom_new_3 = calc_quantile_CRPS(torch.from_numpy(true_target_crps_3),
                                                       torch.from_numpy(samples[:, :, 2, :]),
                                                       torch.from_numpy(eval_points_3))
    denom_3 += denom_new_3

    true_target_crps_6 = np.expand_dims(true_target[:, 5].T, 0)
    eval_points_6 = np.ones_like(true_target_crps_6)
    CRPS_6[test_num], denom_new_6 = calc_quantile_CRPS(torch.from_numpy(true_target_crps_6),
                                                       torch.from_numpy(samples[:, :, 5, :]),
                                                       torch.from_numpy(eval_points_6))
    denom_6 += denom_new_6

    true_target_crps_12 = np.expand_dims(true_target[:, 11].T, 0)
    eval_points_12 = np.ones_like(true_target_crps_12)
    CRPS_12[test_num], denom_new_12 = calc_quantile_CRPS(torch.from_numpy(true_target_crps_12),
                                                         torch.from_numpy(samples[:, :, 11, :]),
                                                         torch.from_numpy(eval_points_12))
    denom_12 += denom_new_12

    test_num = test_num + 1
    if test_num == 1:
        print(f'{test_num} test window out of {test_window} is finished.')
    else:
        print(f'{test_num} test windows out of {test_window} are finished.')

    print('Current RMSE 3, 6, 12 are: {}, {}, {}'.format(np.sqrt(sum(MSE[:, 2]) / test_num),
                                                         np.sqrt(sum(MSE[:, 5]) / test_num),
                                                         np.sqrt(sum(MSE[:, 11]) / test_num)))
    print('Current MAE 3, 6, 12 are: {}, {}, {}'.format(sum(MAE_med[:, 2]) / test_num, sum(MAE_med[:, 5]) / test_num,
                                                        sum(MAE_med[:, 11]) / test_num, ))
    print('Current MAPE 3, 6, 12 are: {}, {}, {}'.format(sum(MAPE_med[:, 2]) / test_num, sum(MAPE_med[:, 5]) / test_num,
                                                         sum(MAPE_med[:, 11]) / test_num, ))
    print('Current CRPS 3, 6, 12 are {}, {}, {}'.format(np.sum(CRPS_3[:test_num]) / denom_3,
                                                        np.sum(CRPS_6[:test_num]) / denom_6,
                                                        np.sum(CRPS_12[:test_num]) / denom_12))

    print('Current average RMSE is {}'.format(np.sqrt(np.mean(MSE[:test_num, :12]))))
    print('Current average MAE is {}'.format(np.mean(MAE_med[:test_num, :12])))
    print('Current average MAPE is {}'.format(np.mean(MAPE_med[:test_num, :12])))
    print('Current overall CRPS is {}'.format(np.sum(CRPS[:test_num]) / denom))

t_end = time.time()
print(f'Sampling time total cost {t_end - t_start}')

