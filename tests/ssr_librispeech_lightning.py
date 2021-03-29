import time
from functools import partial
from pathlib import Path
import os
import yaml
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import pytorch_lightning as pl
from pytorch_lightning.trainer import trainer
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.optim as optim
import torch.utils.data as data

from dataloaders.libri import LibriSpeechDataModule


HOME = Path.home()   # Note, this is not the standard /home/ dir in linux/mac, it is /home/css/ dir
DATA_ROOT = Path.joinpath(HOME, "datasets")
PROJECT_ROOT = Path.joinpath(HOME, "mldl_workout") 
# CWD = Path.cwd()
# the above is useful, BUT gives path only relative to 

# Two ways of getting actual directory for file

# CURRENT_DIR = Path.joinpath(PROJECT_ROOT, "tests", "dataloaders")
CURRENT_DIR = os.path.dirname(__file__)

MODELS_FOLDER = Path.joinpath(PROJECT_ROOT, "models")


train_url = "train-clean-100"
val_url = "dev-clean"
test_url = "test-clean"
# path = "/home/chirantan/datasets"    # GPU machine
path = "/src/datasets"    # docker 
model_path = "/home/chirantan/"
folder = "LibriSpeech"

with open(f'{CURRENT_DIR}/dataloaders/libri_conf.yaml', "r") as conf_file:
    cfg = yaml.safe_load(conf_file)
# https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation

# These can be monitored in comet
h_params = cfg["h_params"]

class LinearPredictor(nn.Module):

    def __init__(self, input_size, output_size=80):
        super(LinearPredictor, self).__init__()
        self.layer = nn.Conv1d(in_channels=input_size, out_channels=output_size,
                               kernel_size=1, stride=1)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, hidden_size)
        inputs = torch.transpose(inputs, 1, 2)
        # inputs: (batch_size, hidden_size, seq_len) -- for conv1d operation

        return torch.transpose(self.layer(inputs), 1, 2)
        # (batch_size, seq_len, output_size) -- back to the original shape


class LSTMEncoder(pl.LightningModule):
    def __init__(self, hyper_params):
        super(LSTMEncoder, self).__init__()

        self.n_feats = hyper_params["n_feats"]   # = mel dims
        self.rnn_dim = hyper_params["rnn_dim"]   # = hidden size
        self.n_rnn_layers = hyper_params["n_rnn_layers"]
        self.rnn_dropout = nn.Dropout(hyper_params["dropout"])
        self.rnn_residual = hyper_params["residual"]

        in_sizes = [self.n_feats] + [self.rnn_dim] * (self.n_rnn_layers - 1)
        out_sizes = [self.rnn_dim] * self.n_rnn_layers
        self.rnns = nn.ModuleList(
            [
                nn.LSTM(input_size=in_size,
                        hidden_size=out_size, batch_first=True)
                for (in_size, out_size) in zip(in_sizes, out_sizes)
            ]
        )

        self.predictnet = LinearPredictor(
            input_size=self.rnn_dim,
            output_size=self.n_feats)

    def forward(self, padded_batch, adj_lengths):
        """Forward function for both training and testing (feature extraction).
        input:
        inputs: (batch_size, seq_len, mel_dim)
        adj_lengths: (batch_size,) -> lengths reduced by min time_shift

        return:
        predicted_mel: (batch_size, seq_len, mel_dim)
        internal_reps: (num_layers + x, batch_size, seq_len, rnn_hidden_size),
            where x is 1 if there's a prenet, otherwise 0
        """
        # seq_len = inputs.size(1)
        max_len = max(adj_lengths)

        # rnn_inputs = inputs
        internal_reps = []

        # packed_rnn_inputs = pack_padded_sequence(rnn_inputs, lengths, True)

        # call pack_pad in forward just before passing to rnn
        packed_rnn_inputs = pack_padded_sequence(padded_batch,
                                                 lengths=adj_lengths.cpu(),
                                                 batch_first=True,
                                                 enforce_sorted=True
                                                 )

        for i, layer in enumerate(self.rnns):
            # print(f"Entering layer {i}")
            packed_rnn_outputs, _ = layer(packed_rnn_inputs)

            rnn_outputs, _ = pad_packed_sequence(
                packed_rnn_outputs, True, total_length=max_len)
            # outputs: (batch_size, seq_len, rnn_hidden_size)

            if i + 1 < len(self.rnns):
                # apply dropout except the last rnn layer
                rnn_outputs = self.rnn_dropout(rnn_outputs)

            rnn_inputs, _ = pad_packed_sequence(
                packed_rnn_inputs, True, total_length=max_len)
            # rnn_inputs: (batch_size, seq_len, rnn_hidden_size)

            if self.rnn_residual and rnn_inputs.size(-1) == rnn_outputs.size(-1):
                # Residual connections
                rnn_outputs = rnn_outputs + rnn_inputs

            internal_reps.append(rnn_outputs)

            packed_rnn_inputs = pack_padded_sequence(
                rnn_outputs, adj_lengths.cpu(), True)
            # print(f"Done with layer {i}")

        predicted_logmel_frame = self.predictnet(rnn_outputs)
        # predicted_mel: (batch_size, seq_len, mel_dim)

        internal_reps = torch.stack(internal_reps)

        return predicted_logmel_frame, internal_reps

    def training_step(self, batch, batch_idx):
        # model.train()

        train_losses = []

        criterion = nn.L1Loss()
        time_shifts = sorted(h_params["time_shifts"])
        # scaler = torch.cuda.amp.GradScaler()

        # with experiment.train():
        orig_start = time.time()
        # for batch_num, (batch, lenghts) in enumerate(train_loader):
        batch, lenghts = batch
        if batch_idx == 0:
            prev = orig_start
        else:
            prev = time.time()
        # torch.cuda.empty_cache()
        # optimizer.zero_grad()
        # batch = batch.to(device)
        # print(batch.shape)
        # with torch.cuda.amp.autocast():
        # outputs, _ = model(batch[:, :-time_shifts[0], :], lenghts-time_shifts[0])
        outputs, _ = self(
            batch[:, :-time_shifts[0], :], lenghts-time_shifts[0])

    # print(f"train outputs type: {type(outputs)}")

        # ts_loss_tensor = torch.empty(3).to(device)
        ts_loss_tensor = torch.empty(3).type_as(outputs)
        # ts_loss_tensor = ts_loss_tensor.type_as(outputs)

        for i, time_shift in enumerate(time_shifts):
            if i == 0:
                pred_spec = outputs  # .to("cpu")
            else:
                # .to("cpu")
                pred_spec = outputs[:, :-(time_shift-time_shifts[0])]

            # loss = criterion(pred_spec, batch[:, time_shift:, :]).to(device)
            loss = criterion(pred_spec, batch[:, time_shift:, :])
            ts_loss_tensor[i] = loss

        mean_loss = torch.mean(ts_loss_tensor)
        # train_losses.append(ts_loss_tensor)

        # for i, ts in enumerate(time_shifts):
        #     # experiment.log_metric(f"loss_{ts}", loss_list[i].item(), step=global_step, epoch=epoch, include_context=True)
        #     experiment.log_metric(f"loss_{ts}", ts_loss_tensor[i].item(), step=global_step, epoch=epoch, include_context=True)

        # experiment.log_metric("batch_loss", batch_loss, step=global_step, epoch=epoch, include_context=True)

        # scaler.scale(batch_loss).backward()

        # grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(),
        #                                             hyper_params["clip_threshold"])
        # optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()

        # print(
        #     f"epoch time: {(time.time() - orig_start):.5f}, iter time: {(time.time()-prev):.3f}")

        # experiment.log_metric("gradient norm", grad_norm, step=global_step, epoch=epoch, include_context=True)

        # global_step += 1

        # lets lightning know this is the loss it needs to optimize
        return {'loss': mean_loss}

    def validation_step(self, batch, batch_idx):
        val_losses = []
        criterion = nn.L1Loss()
        time_shifts = sorted(h_params["time_shifts"])
        batch, lenghts = batch
        # with experiment.validate():
        # outputs, _ = model(val_batch[:, :-time_shifts[0], :], val_batch_lenghts-time_shifts[0])
        outputs, _ = self(batch[:, :-time_shifts[0], :],
                          lenghts-time_shifts[0])

        # print(f"val output type: {type(outputs)}")

        val_ts_loss_tensor = torch.empty(3).type_as(outputs)
        for i, time_shift in enumerate(time_shifts):
            if i == 0:
                val_pred_spec = outputs  # .to("cpu")
            else:
                # .to("cpu")
                val_pred_spec = outputs[:, :-(time_shift-time_shifts[0])]

            val_loss = criterion(val_pred_spec, batch[:, time_shift:, :])

            # val_ts_loss_tensor[i] = val_loss.item()
            val_ts_loss_tensor[i] = val_loss

            # val_losses.append(val_ts_loss_tensor)
        val_mean_loss = torch.mean(val_ts_loss_tensor)
        return {'val_loss': val_mean_loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=h_params["learning_rate"])
        return optimizer


def main():

    # use_cuda = torch.cuda.is_available()
    # # device = torch.device("cuda" if use_cuda else "cpu")
    # print(f"Train dataset length: {len(train_dataset)}")
    # print(f"Val dataset length: {len(val_dataset)}")
    # print(f"Test dataset length: {len(test_dataset)}")

    # global_step = 1   # total batches, across epochs etc.
    lae = LSTMEncoder(h_params)
    libri_dm = LibriSpeechDataModule() 

    stage = ''   # debug or anything else

    if stage == 'debug':
        trainer = pl.Trainer(fast_dev_run=True, gpus=1)
    else:
        trainer = pl.Trainer(max_epochs=100,
                            gradient_clip_val=h_params["clip_threshold"],
                            gpus=1,
                            #  check_val_every_n_epoch=1,
                            accumulate_grad_batches=5,
                            val_check_interval=0.5,
                            limit_train_batches=1.0,
                            limit_val_batches=1.0,
                            limit_test_batches=1.0,
                            precision=16,
                            )

    trainer.fit(lae,
                libri_dm,
                # train_dataloader=train_loader,
                # val_dataloaders=val_loader,
                )


if __name__ == "__main__":
    main()
