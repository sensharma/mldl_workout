import os
import time
from functools import partial

from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
# torchaudio.set_audio_backend("soundfile")
import numpy as np

comet_experiment = "test"   # use empty string "" to deactivate comet logging

# Adjust path and folder as required
train_url = "train-clean-100"
test_url = "test-clean"
# path = "/Users/chirantan/Datasets/"
path = "./data/"    # GPU machine
folder = "LibriSpeech"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

hyper_params = {
    "train_data_limit": 1000,  # Use None to train on full dataset, 100 for test
    "n_rnn_layers": 3,
    "random": False,
    "bi_dir": False,
    "rnn_dim": 512,  # hidden_size param
    "n_feats": 80,  # input_size param (no. of mel filters)
    "dropout": 0.1,
    "residual": True,
    "optimizer": "adam",
    "time_shifts": [2, 5, 8],  #[2, 5, 8]: n: predict f_{t + n} | f_{t} (2, 5, 10, 20 used in paper)
    "clip_threshold": 5.0,
    "learning_rate": 1e-4,
    "batch_size": 64,  # (64 works on 8GB GPU)
    "n_epochs": 2,  # 100 in paper
}

params = {
    "data_loader_args": {"num_workers": 4, "pin_memory": True} # 4 seems to be optimal
                        if use_cuda else {"num_workers": 2},
    "model_path": "./models",
}

if comet_experiment:
    experiment = Experiment(workspace="ssr-pytorch", 
                            project_name="kth-ssr-project")
    experiment.set_name(comet_experiment)
    if experiment.alive is False:
        raise Exception("Comet experiment not working")
else:
  experiment = Experiment(api_key='dummy_key', disabled=True)

experiment.log_parameters(hyper_params)

if use_cuda:
    if hyper_params["random"]:
        torch.backends.cudnn.benchmark = True
    else:
        seed = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
else:
    if hyper_params["random"]:
        torch.backends.mkldnn.benchmark = True


# Set download to True if download also required
train_dataset = torchaudio.datasets.LIBRISPEECH(
    path, folder_in_archive=folder, url=train_url, download=False
)
test_dataset = torchaudio.datasets.LIBRISPEECH(
    path, folder_in_archive=folder, url=test_url, download=False
)



def pre_processing(data, transform="logmel"):
    """[summary]

    # create list out of data batch 
    # sort
    # get lengths
    # pad sorted (also coverts to torch tensor)
    # pack padded sequence

    Args:
        data ([type]): [description]
        transform (str, optional): [description]. Defaults to "logmel".

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """
    logmel_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, n_mels=80  # def = 400  # def = 128
        ),
        torchaudio.transforms.AmplitudeToDB(),  # for log mel
    )

    mask_logmel_transforms = nn.Sequential(
        logmel_transforms,
        torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
        torchaudio.transforms.TimeMasking(time_mask_param=100),
    )

    # change for validation if changing here
    valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

    spectrograms = []
    # input_lengths = []
    # for (waveform, _, utterance, _, _, _) in data:
    for (waveform, _, _, _, _, _) in data:
        if transform == "logmel":
            spec = logmel_transforms(waveform).squeeze(0).transpose(0, 1)
        elif transform == "mask_norm_logmel":
            spec = mask_logmel_transforms(waveform).squeeze(0).transpose(0, 1)
        elif transform == "valid":
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception(
                "transform should be logmel, mask_norm_logmel or valid"
            )
        spectrograms.append(spec)

    # for spec in spectrograms:
    #     print(spec.shape)

    sorted_specs = sorted(spectrograms, key=lambda spec: spec.shape[0], reverse=True)
    lengths = torch.tensor([spec.shape[0] for spec in sorted_specs])
    padded_batch = nn.utils.rnn.pad_sequence(sorted_specs, batch_first=True)

    # print(padded_batch_tensor.shape)
    # for item in padded_batch_tensor:
    #     print(item.shape)

    # packed_batch = pack_padded_sequence(
    #     padded_batch, lengths=lengths, batch_first=True, enforce_sorted=True
    # )

    # both the returned items are torch tensors
    return padded_batch, lengths


if hyper_params["train_data_limit"] is None:
    use_train_dataset = train_dataset
else:
    use_train_dataset = torch.utils.data.Subset(
        train_dataset, range(0, hyper_params["train_data_limit"])
    )

train_loader = data.DataLoader(
    dataset=use_train_dataset,
    batch_size=hyper_params["batch_size"],
    shuffle=True,
    # collate_fn=lambda x: pre_processing(x, "logmel"),
    collate_fn=partial(pre_processing, transform="logmel"),
    **params["data_loader_args"]
)

test_loader = data.DataLoader(
    dataset=test_dataset,
    batch_size=hyper_params["batch_size"],
    shuffle=False,
    # collate_fn=lambda x: pre_processing(x, "valid"),
    collate_fn=partial(pre_processing, transform="valid"),
    **params["data_loader_args"]
)

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

class LSTMEncoder(nn.Module):
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
                nn.LSTM(input_size=in_size, hidden_size=out_size, batch_first=True)
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
                                            lengths=adj_lengths, 
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

            packed_rnn_inputs = pack_padded_sequence(rnn_outputs, adj_lengths, True)
            # print(f"Done with layer {i}")

        predicted_logmel_frame = self.predictnet(rnn_outputs)
        # predicted_mel: (batch_size, seq_len, mel_dim)

        internal_reps = torch.stack(internal_reps)

        return predicted_logmel_frame, internal_reps
        # predicted_mel is only for training; internal_reps is the extracted
        # features

        # self.rnn_stack = nn.LSTM(input_size=self.n_feats,
        #                          hidden_size=self.rnn_dim,
        #                          num_layers=self.n_rnn_layers,
        #                          batch_first=True,
        #                          dropout=self.rnn_dropout)

# Training loop
# wrapped in if __name__ .... for multiple dataloader worker processes to run safely
# https://pytorch.org/docs/stable/data.html#multi-process-data-loading
 

def train():
    model = LSTMEncoder(hyper_params).to(device)

    criterion = nn.L1Loss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyper_params["learning_rate"])

    global_step = 1   # total iters, i.e, 1 epoch and 2 batches = 102, etc.
    for epoch in range(hyper_params["n_epochs"]):
        print(f"\n Epoch: {epoch} \n")

        ####################
        ##### Training #####
        ####################

        model.train()
        train_losses = []
        time_shifts = sorted(hyper_params["time_shifts"])

        orig_start = time.time()
        with experiment.train():
            for batch_num, (batch, lenghts) in enumerate(train_loader):
                if batch_num == 0:
                    prev = orig_start
                else:
                    prev = time.time()
                batch = batch.to(device)
                # print(batch)
                # if batch_num == 2:
                #     break

        # torch.save(model.state_dict(),
        #     open(os.path.join(model_dir, config.experiment_name + '__epoch_0.model'),
        #     'wb'))

                # print(batch.shape)
                # quit()

                outputs, _ = model(
                batch[:, :-time_shifts[0], :], lenghts-time_shifts[0])

                # print(outputs.shape)

                optimizer.zero_grad()

                """
                for i, t_shift in enumerate(time_shifts):
                # independent of t_shift
                in_spec = orig_spec[:, :-time_shifts[0]]

                # dependent on t_shift
                if i == 0:
                    pred_spec = out_spec
                else:
                    pred_spec = out_spec[:, :-(t_shift-time_shifts[0])]
                target_spec = orig_spec[:, t_shift:]

                losses.append(criterion(pred_spec, target_spec).item())
                """

                loss_list = []       # list of losses computed for no. of time_shifts being used
                for i, time_shift in enumerate(time_shifts):
                    if i==0:
                        pred_spec = outputs
                    else:
                        pred_spec = outputs[:, :-(time_shift-time_shifts[0])]

                    loss = criterion(pred_spec, batch[:, time_shift:, :]).to(device)
                    loss_list.append(loss)

                batch_loss = sum(loss_list)

                # train_losses.append(loss.item())
                train_losses.append([l.item() for l in loss_list])

                for i, ts in enumerate(time_shifts): 
                    experiment.log_parameter(f"loss_{ts}", loss_list[i].item())

                experiment.log_parameter("batch_loss", batch_loss)

                batch_loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                            hyper_params["clip_threshold"])
                optimizer.step()

                print(f"epoch time: {(time.time() - orig_start):.5f}, iter time: {(time.time()-prev):.3f}")
                # log_value("training loss (step-wise)", float(loss.item()), global_step)
                experiment.log_parameter("gradient norm", grad_norm, global_step)

                global_step += 1

        ######################
        ##### Validation #####
        ######################

        # model.eval()
        # val_losses = []
        # with experiment.test():
        #     with torch.set_grad_enabled(False):
        #         for val_batch_x, val_batch_l in val_data_loader:
        #             _, val_indices = torch.sort(val_batch_l, descending=True)

        #             val_batch_x = Variable(val_batch_x[val_indices]).cuda()
        #             val_batch_l = Variable(val_batch_l[val_indices]).cuda()

        #             val_outputs, _ = model(
        #                 val_batch_x[:, :-config.time_shift, :],
        #                 val_batch_l - config.time_shift)

        #             val_loss = criterion(val_outputs,
        #                                     val_batch_x[:, config.time_shift:, :])
        #             val_losses.append(val_loss.item())

        #     logging.info('Epoch: %d Training Loss: %.5f Validation Loss: %.5f' % (epoch_i + 1, np.mean(train_losses), np.mean(val_losses)))

        #     log_value("training loss (epoch-wise)", np.mean(train_losses), epoch_i)
        #     log_value("validation loss (epoch-wise)", np.mean(val_losses), epoch_i)

        #     torch.save(model.state_dict(),
        #         open(os.path.join(model_dir, config.experiment_name + '__epoch_%d' % (epoch_i + 1) + '.model'), 'wb'))

if __name__ == "__main__":
    train()