import os
import time
from typing import List, Optional, Union
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import numpy as np
import random
from gluonts.core.component import validated
from .pytorchtools import EarlyStopping
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class Trainer:
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        num_batches_per_epoch_val: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        maximum_learning_rate: float = 1e-2,
        clip_gradient: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        patience: int = 5,
        path: str = "model.pt",
        path_rng: str = "rng.pt",
        load_model: bool = True,
        seed: int = 777,
        mean: float = None,
        std: float = None,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_batches_per_epoch_val = num_batches_per_epoch_val
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.clip_gradient = clip_gradient
        self.device = device
        self.patience = patience
        self.path = path
        self.path_rng = path_rng
        self.load_model = load_model
        self.seed = seed
        self.mean = mean
        self.std = std

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:

        # Check if the model is already trained with current configuration
        # Then decide if you want to retrain or start sampling
        if self.load_model:
            if os.path.exists(self.path):
                reply = str(input('Model with the current configuration is already saved.'
                                  'Do you want to rewrite the saved model? (y/n)')).lower().strip()
                if reply[0] == 'y':
                    return
                else:
                    rng = torch.load(self.path_rng)
                    torch.set_rng_state(rng)
                    net.load_state_dict(torch.load(self.path))
                    net.eval()
                    print('Start sampling with the saved model...')
                    return

        optimizer = Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )

        early_stopping = EarlyStopping(patience=self.patience, verbose= True, path = self.path, path_rng = self.path_rng)

        for epoch_no in range(self.epochs):

            set_seed(self.seed)

            cumm_epoch_loss = 0.0
            total = self.num_batches_per_epoch - 1
            total_val = self.num_batches_per_epoch_val - 1

            # training loop
            loss_train = []
            with tqdm(train_iter, total=total, colour= "red") as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()

                    inputs_train = [v.to(self.device) for v in data_entry.values()]
                    output = net(*inputs_train)

                    if isinstance(output, (list, tuple)):
                        loss = output[0]
                    else:
                        loss = output

                    loss_train.append(loss)
                    cumm_epoch_loss += loss.item()
                    avg_epoch_loss = cumm_epoch_loss / batch_no
                    it.set_postfix(
                        {
                            "epoch": f"{epoch_no + 1}/{self.epochs}",
                            "avg_train_loss": avg_epoch_loss,
                        },
                        refresh=False,
                    )

                    loss.backward()
                    if self.clip_gradient is not None:
                        nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)

                    optimizer.step()
                    lr_scheduler.step()

                    if self.num_batches_per_epoch == batch_no:
                        break
                train_loss = loss.item()
                it.close()

            # validation loop
            loss_val = []
            if validation_iter is not None:
                cumm_epoch_loss_val = 0.0
                with tqdm(validation_iter, total = total_val, colour="green") as it:
                    for batch_no, data_entry in enumerate(it, start=1):
                        inputs_val = [v.to(self.device) for v in data_entry.values()]
                        with torch.no_grad():
                            output = net(*inputs_val)
                        if isinstance(output, (list, tuple)):
                            loss = output[0]
                        else:
                            loss = output

                        loss_val.append(loss)
                        cumm_epoch_loss_val += loss.item()
                        avg_epoch_loss_val = cumm_epoch_loss_val / batch_no
                        it.set_postfix(
                            {
                                "epoch": f"{epoch_no + 1}/{self.epochs}",
                                "avg_val_loss": avg_epoch_loss_val,
                            },
                            refresh=False,
                        )

                        if self.num_batches_per_epoch_val == batch_no:
                            break
                    it.close()

                    # Early stopping, load the model with the lowest validation loss
                    early_stopping(avg_epoch_loss_val, net)

                    if early_stopping.early_stop:
                        net.load_state_dict(torch.load(self.path))
                        net.eval()
                        print('Early stopping')
                        break

                    # Load the best model if last epoch is in early stopping counter
                    if epoch_no == self.epochs - 1:
                        net.load_state_dict(torch.load(self.path))
                        net.eval()





            
