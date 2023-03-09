import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from graphnet.models.graph_builders import KNNGraphBuilder

from model import IceCubeModel
from preprocessing import prepare_sensors
from datasets import IceCubeSubmissionDatase

from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import ModelCheckpoint


# CHECKPOINT_CALLBACK = ModelCheckpoint(dirpath='./models/', save_top_k=1, monitor="val_loss")
# NUM_EPOCHS = CFG.num_epochs


# class TTAWrapper(nn.Module):
#     def __init__(
#         self,
#         model,
#         device,
#         angles=[0, 180],
#     ):
#         super().__init__()
#         self.model = model
#         self.device = device
#         self.angles = [a * np.pi / 180 for a in angles]
#         self.rmats = [self.rotz(a) for a in self.angles]

#     def rotz(self, theta):
#         # Counter clockwise rotation
#         return (
#             torch.tensor(
#                 [
#                     [np.cos(theta), -np.sin(theta), 0],
#                     [np.sin(theta), np.cos(theta), 0],
#                     [0, 0, 1],
#                 ],
#                 dtype=torch.float32,
#             )
#             .unsqueeze(0)
#             .to(self.device)
#         )

#     def forward(self, data):
#         azi_out_sin, azi_out_cos, zen_out = 0, 0, 0
#         data_rot = data

#         for a, mat in zip(self.angles, self.rmats):
#             data_rot.x[:, :3] = torch.matmul(data.x[:, :3], mat)
#             a_out, z_out = self.model(data_rot)

#             # Remove rotation from the azimuth prediction
#             azi_out_sin += torch.sin(a_out + a)
#             azi_out_cos += torch.cos(a_out + a)
#             zen_out += z_out

#         # https://en.wikipedia.org/wiki/Circular_mean
#         azi_out = torch.atan2(azi_out_sin, azi_out_cos)
#         zen_out /= len(self.angles)

#         return azi_out, zen_out



def train_fn(dataset_path, batch_size=32, device="cuda", num_epochs):
    model = IceCubeModel

    train_dataset, val_dataset = # func (dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
    trainer = Trainer(default_root_dir=MODEL_CACHE, devices=device, max_epochs=num_epochs, callbacks=[CHECKPOINT_CALLBACK])
    trainer.fit(model, train_loader, val_loader)

    return None

def make_train(dataset_paths, device="cuda", suffix="metric", mode="train"):
    mpaths = []
    for p in dataset_paths:
        train_fn(dataset_path=p, num_epochs=NUM_EPOCHS)

    num_models = len([item for sublist in mpaths for item in sublist])
    print(f"{num_models} models found.")

    

# def make_predictions(dataset_paths, device="cuda", suffix="metric", mode="test"):
#     mpaths = []
#     for p in dataset_paths:
#         mpaths.append(sorted(list(p.rglob(f"*{suffix}.ckpt"))))

#     num_models = len([item for sublist in mpaths for item in sublist])
#     print(f"{num_models} models found.")

#     sensors = prepare_sensors()
#     # sensors["sensor_id"] = sensors["sensor_id"].astype(np.int16)
#     # sensors = pls.from_pandas(sensors)

#     meta = pd.read_parquet(
#         INPUT_PATH / f"{mode}_meta.parquet", columns=["batch_id", "event_id"]
#     ).astype(_dtype)
#     batch_ids = meta["batch_id"].unique()
#     output = 0

#     if mode == "train":
#         batch_ids = batch_ids[:6]

#     # for i, group in enumerate(mpaths):
#     #     for j, p in enumerate(group):

#     p = mpaths[0][0]
#     model = IceCubeModel.load_from_checkpoint(p, strict=False)
#     pre_transform = KNNGraphBuilder(nb_nearest_neighbours=8)

#     batch_preds = []
#     for b in batch_ids:
#         event_ids = meta[meta["batch_id"] == b]["event_id"].tolist()
#         dataset = IceCubeSubmissionDataset(
#             b, event_ids, sensors, mode=mode, pre_transform=pre_transform
#         )
#         batch_preds.append(infer(model, dataset, device=device, batch_size=1024))
#         print("Finished batch", b)

#         if mode == "train" and b == 6:
#             break

#     output += torch.cat(batch_preds, 0)

#     # After looping through folds
#     output /= num_models

#     event_id_labels = []
#     for b in batch_ids:
#         event_id_labels.extend(meta[meta["batch_id"] == b]["event_id"].tolist())

#     sub = {
#         "event_id": event_id_labels,
#         "azimuth": output[:, 0],
#         "zenith": output[:, 1],
#     }

#     sub = pd.DataFrame(sub)
#     sub.to_csv("submission.csv", index=False)
