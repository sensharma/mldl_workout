#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from comet_ml import Experiment


# In[2]:


import os
import time
from functools import partial
import gc
from statistics import mean

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
torchaudio.set_audio_backend("sox_io") 


# In[3]:


train_url = "train-clean-100"
val_url = "dev-clean"
test_url = "test-clean"
path = "/home/chirantan/datasets"    # GPU machine
model_path = "/home/chirantan/"
folder = "LibriSpeech"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

params = {
    "data_loader_args": {"num_workers": 6, "pin_memory": True} # 4 seems to be optimal
                        if use_cuda else {"num_workers": 2},
}

print(f"device: {device}")


# In[4]:


# These will be monitored in comet
hyper_params = {
    "train_data_limit": None,  # Use None to train on full dataset, 100 etc. for test
    "val_data_limit": None,    # Use None to train on full dataset, 100 etc. for test
    "n_rnn_layers": 4,
    "random": False,
    "bi_dir": False,
    "rnn_dim": 512,  # hidden_size param
    "n_feats": 80,  # input_size param (no. of mel filters)
    "dropout": 0.1,
    "residual": True,
    "optimizer": "adam",
    "time_shifts": [2, 5, 8],  #[2, 5, 8]: n: predict f_{t + n} | f_{t} (2, 5, 10, 20 used in paper)
    "clip_threshold": 1.0,
    "learning_rate": 1e-4,
    "batch_size": 64,  # (8-12G(gtx 1070, K80, T4): 64; 16G(P100): 128; 16G(V100): 256?)
    "n_epochs": 2,  # 100 in paper
}


# In[5]:


train_dataset = torchaudio.datasets.LIBRISPEECH(
    path, folder_in_archive=folder, url=train_url, download=True
)
val_dataset = torchaudio.datasets.LIBRISPEECH(
    path, folder_in_archive=folder, url=val_url, download=True
)
test_dataset = torchaudio.datasets.LIBRISPEECH(
    path, folder_in_archive=folder, url=test_url, download=True
)


# In[6]:


print(f"Train dataset length: {len(train_dataset)}")
print(f"Val dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")


# In[7]:


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

comet_api_key = "TaZfBlTz3126asNAf3liOlp5l"
# comet_experiment = "T4_2ep_4GRU_test"   # use empty string "" to deactivate comet logging
comet_experiment = ""   # use empty string "" to deactivate comet logging

if comet_experiment:
    experiment = Experiment(api_key=comet_api_key, 
                            workspace="ssr-pytorch", 
                            project_name="kth-ssr-project")
    experiment.set_name(comet_experiment)
    if experiment.alive is False:
        raise Exception("Comet experiment not working")
else:
  experiment = Experiment(api_key='dummy_key', disabled=True)

experiment.log_parameters(hyper_params)


# In[8]:


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

    # At the moment same as training, but keep validation separate.
    # Is required if doing masking etc. for training
    valid_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, n_mels=80  # def = 400  # def = 128
        ),
        torchaudio.transforms.AmplitudeToDB(),  # for log mel
    )

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

    sorted_specs = sorted(spectrograms, key=lambda spec: spec.shape[0], reverse=True)
    lengths = torch.tensor([spec.shape[0] for spec in sorted_specs])
    padded_batch = nn.utils.rnn.pad_sequence(sorted_specs, batch_first=True)

    # both the returned items are torch tensors
    return padded_batch, lengths


# In[9]:


if hyper_params["train_data_limit"] is None:
    use_train_dataset = train_dataset
else:
    use_train_dataset = torch.utils.data.Subset(
        train_dataset, range(0, hyper_params["train_data_limit"])
    )

if hyper_params["val_data_limit"] is None:
    use_val_dataset = val_dataset
else:
    use_val_dataset = torch.utils.data.Subset(
        val_dataset, range(0, hyper_params["val_data_limit"])
    )

train_loader = data.DataLoader(
    dataset=use_train_dataset,
    batch_size=hyper_params["batch_size"],
    shuffle=True,
    # collate_fn=lambda x: pre_processing(x, "logmel"),
    collate_fn=partial(pre_processing, transform="logmel"),
    **params["data_loader_args"]
)

val_loader = data.DataLoader(
    dataset=use_val_dataset,
    batch_size=hyper_params["batch_size"],
    shuffle=False,
    # collate_fn=lambda x: pre_processing(x, "valid"),
    collate_fn=partial(pre_processing, transform="valid"),
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


# In[10]:


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


# In[11]:


def train(model, criterion, time_shifts, epoch, global_step):
    
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=hyper_params["learning_rate"])

    train_losses = []

    scaler = torch.cuda.amp.GradScaler()

    with experiment.train():
        orig_start = time.time()
        for batch_num, (batch, lenghts) in enumerate(train_loader):
            if batch_num == 0:
                prev = orig_start
            else:
                prev = time.time()
            # torch.cuda.empty_cache()
            optimizer.zero_grad()
            batch = batch.to(device)
            print(batch.shape)
            with torch.cuda.amp.autocast():
                outputs, _ = model(batch[:, :-time_shifts[0], :], lenghts-time_shifts[0])
            
            # print(f"train outputs type: {type(outputs)}")

                ts_loss_tensor = torch.empty(3).to(device)
                for i, time_shift in enumerate(time_shifts):
                    if i==0:
                        pred_spec = outputs   #.to("cpu")
                    else:
                        pred_spec = outputs[:, :-(time_shift-time_shifts[0])]   #.to("cpu")

                    # loss = criterion(pred_spec, batch[:, time_shift:, :]).to(device)
                    loss = criterion(pred_spec, batch[:, time_shift:, :])
                    ts_loss_tensor[i] = loss

                batch_loss = torch.mean(ts_loss_tensor)
                train_losses.append(ts_loss_tensor)

            for i, ts in enumerate(time_shifts):
                # experiment.log_metric(f"loss_{ts}", loss_list[i].item(), step=global_step, epoch=epoch, include_context=True)
                experiment.log_metric(f"loss_{ts}", ts_loss_tensor[i].item(), step=global_step, epoch=epoch, include_context=True)

            experiment.log_metric("batch_loss", batch_loss, step=global_step, epoch=epoch, include_context=True)

            scaler.scale(batch_loss).backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                        hyper_params["clip_threshold"])
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            print(f"epoch time: {(time.time() - orig_start):.5f}, iter time: {(time.time()-prev):.3f}")
    
            experiment.log_metric("gradient norm", grad_norm, step=global_step, epoch=epoch, include_context=True)

            global_step += 1

    return train_losses


# In[12]:


def validate(model, criterion, time_shifts):
    model.eval()
    val_start = time.time()
    val_losses = []
    # with experiment.validate():
    with experiment.context_manager("valid"):
        with torch.set_grad_enabled(False):
            for val_batch_num, (val_batch, val_batch_lenghts) in enumerate(val_loader):
                # torch.cuda.empty_cache()
                val_batch = val_batch.to(device)
                # print(f"val batch shape: {val_batch.shape}")
                outputs, _ = model(val_batch[:, :-time_shifts[0], :], val_batch_lenghts-time_shifts[0])

                # print(f"val output type: {type(outputs)}")

                val_ts_loss_tensor = torch.empty(3)
                for i, time_shift in enumerate(time_shifts):
                    if i==0:
                        val_pred_spec = outputs   #.to("cpu")
                    else:
                        val_pred_spec = outputs[:, :-(time_shift-time_shifts[0])]   #.to("cpu")

                    val_loss = criterion(val_pred_spec, val_batch[:, time_shift:, :])

                    val_ts_loss_tensor[i] = val_loss.item()

                val_losses.append(val_ts_loss_tensor)
    print(f"Val time: {time.time() - val_start}")
    return val_losses


# In[13]:


def main():
    model = LSTMEncoder(hyper_params).to(device)
    time_shifts = sorted(hyper_params["time_shifts"])
    criterion = nn.L1Loss().to(device)


    global_step = 1   # total batches, across epochs etc.
    for epoch in range(hyper_params["n_epochs"]):
        print(f"\n Epoch: {epoch} \n")

        train_losses = train(model, criterion, time_shifts, epoch, global_step)
        val_losses = validate(model, criterion, time_shifts)

        train_losses_tensor = torch.stack(train_losses)
        train_epoch_means_tensor = torch.mean(train_losses_tensor, dim=0)

        val_losses_tensor = torch.stack(val_losses)
        val_epoch_means_tensor = torch.mean(val_losses_tensor, dim=0)

        for i, ts in enumerate(time_shifts):
            experiment.log_metrics({f"train_epoch_avg_loss_{ts}":  train_epoch_means_tensor[i].item(), 
                                    f"val_epoch_avg_loss_{ts}":  val_epoch_means_tensor[i].item()},
                                    step=epoch, 
                                    epoch=epoch)
        
        experiment.log_metrics({f"train_epoch_avg_loss_tot": torch.mean(train_epoch_means_tensor).item(),
                                f"val_epoch_avg_loss_tot": torch.mean(val_epoch_means_tensor).item()},
                                step=epoch, 
                                epoch=epoch)
    experiment.end()
    print("Run completed!")


# In[15]:


# gc.collect()
# torch.cuda.empty_cache()


# In[14]:

if __name__ == '__main__':
    main()

# In[ ]:




