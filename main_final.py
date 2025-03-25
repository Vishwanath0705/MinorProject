import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from evaluate import load
import numpy as np

from typing import List, Dict

import dataloading
import model_define

pl.seed_everything(445326, workers=True)

data = dataloading.YelpDataLoader()
data.prepare_data()

data.setup()
print(len(data.train))
print(len(data.val))
print(len(data.test))

from model_define import Model
model = Model()

MAX_EPOCHS = 15

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    dirpath="model",
    filename="yelp-sentiment-multilingual-{epoch:02d}-{val_loss:.3f}",
    save_top_k=3,
    mode="min")

trainer = pl.Trainer(max_epochs=MAX_EPOCHS, 
                     callbacks=[checkpoint_callback])

trainer.fit(model, data.train_dataloader(), data.val_dataloader())

trainer.test(dataloaders=data.test_dataloader())

torch.save(model.state_dict(), "final_model.pth")
import torch
from model_define import Model  

model = Model()

model.load_state_dict(torch.load("final_model.pth"))

model.eval()

print("Model loaded successfully!")
