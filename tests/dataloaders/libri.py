from pathlib import Path
from functools import partial
import yaml
import os

import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl

import torchaudio
torchaudio.set_audio_backend("sox_io")
from torchaudio.datasets import LIBRISPEECH
# https://github.com/pytorch/audio/issues/903

HOME = Path.home()   # Note, this is not the standard /home/ dir in linux/mac, it is /home/css/ dir
DATA_ROOT = Path.joinpath(HOME, "datasets")
PROJECT_ROOT = Path.joinpath(HOME, "mldl_workout") 
# CWD = Path.cwd()  
# the above is useful, BUT gives path only relative to 

# Two ways of getting actual directory for file

# CURRENT_DIR = Path.joinpath(PROJECT_ROOT, "tests", "dataloaders")
CURRENT_DIR = os.path.dirname(__file__)

MODELS_FOLDER = Path.joinpath(PROJECT_ROOT, "models")
DATA_FOLDER = "LibriSpeech"
print(CURRENT_DIR)

# with open("./libri_conf.yaml", "r") as conf_file:
with open(f'{CURRENT_DIR}/libri_conf.yaml', "r") as conf_file:
    cfg = yaml.safe_load(conf_file)
# https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation

h_params = cfg["h_params"]
dl_params = cfg["dataloader_params"]
ds_params = cfg["dataset_params"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
pl.LightningDataModule typical methods:https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html?highlight=lightningdatamodule
- init
- prepare_data(called from 1 GPU): download etc. 
- setup(called from all GPUs): paths, train/val/test splits etc.
- additional preprocessing funcs (if only simple pre-processing, include in setup)
- train_dataloader
- val_dataloader
- test_dataloader
"""

class LibriSpeechDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=DATA_ROOT, batch_size=h_params['batch_size']):

        super().__init__()

        self.datadir = data_dir
        self.batch_size = batch_size
        if torch.cuda.is_available():
            self.params = cfg['dataloader_params']
        else:
            self.params = {"num_workers": 2}
        # self.params = {
        #     # 4 seems to be optimal
        #     "data_loader_args": {"num_workers": 6, "pin_memory": True}
        #     if torch.cuda.is_available() else {"num_workers": 2},
        # }
        # hardcoding dataset specific information if required

    def prepare_data(self):
        # download
        # prepare_data is called from a single GPU.
        # Do not use it to assign state (self.x = y).

        LIBRISPEECH(
            self.datadir, folder_in_archive=DATA_FOLDER, url=ds_params["train_url"], download=True
        )
        LIBRISPEECH(
            self.datadir, folder_in_archive=DATA_FOLDER, url=ds_params["val_url"], download=True
        )
        LIBRISPEECH(
            self.datadir, folder_in_archive=DATA_FOLDER, url=ds_params["test_url"], download=True
        )

    def setup(self, stage=None):
        #  setup is called from every GPU. Setting state here is okay.
        if stage == 'fit' or stage is None:
            self.libri_train = LIBRISPEECH(self.datadir,
                                           folder_in_archive=DATA_FOLDER,
                                           url=ds_params["train_url"],
                                           )

            self.libri_val = LIBRISPEECH(self.datadir,
                                         folder_in_archive=DATA_FOLDER,
                                         url=ds_params["val_url"],
                                         )

        if stage == 'test' or stage is None:
            self.libri_test = LIBRISPEECH(self.datadir,
                                          folder_in_archive=DATA_FOLDER,
                                          url=ds_params["test_url"],
                                          )

    @staticmethod
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
                spec = mask_logmel_transforms(
                    waveform).squeeze(0).transpose(0, 1)
            elif transform == "valid":
                spec = valid_audio_transforms(
                    waveform).squeeze(0).transpose(0, 1)
            else:
                raise Exception(
                    "transform should be logmel, mask_norm_logmel or valid"
                )
            spectrograms.append(spec)

        sorted_specs = sorted(
            spectrograms, key=lambda spec: spec.shape[0], reverse=True)
        lengths = torch.tensor([spec.shape[0] for spec in sorted_specs])
        padded_batch = nn.utils.rnn.pad_sequence(
            sorted_specs, batch_first=True)

        # both the returned items are torch tensors
        return padded_batch, lengths

    def train_dataloader(self):
        train_loader = data.DataLoader(
            dataset=self.libri_train,
            batch_size=h_params["batch_size"],
            shuffle=True,
            # collate_fn=lambda x: pre_processing(x, "logmel"),
            collate_fn=partial(self.pre_processing, transform="logmel"),
            **self.params
        )
        return train_loader

    def val_dataloader(self):
        val_loader = data.DataLoader(
            dataset=self.libri_val,
            batch_size=h_params["batch_size"],
            shuffle=False,
            # collate_fn=lambda x: pre_processing(x, "valid"),
            collate_fn=partial(self.pre_processing, transform="valid"),
            **self.params
        )
        return val_loader

    def test_dataloader(self):
        test_loader = data.DataLoader(
            dataset=self.libri_test,
            batch_size=h_params["batch_size"],
            shuffle=False,
            # collate_fn=lambda x: pre_processing(x, "valid"),
            collate_fn=partial(self.pre_processing, transform="valid"),
            **self.params
        )
        return test_loader
