# -*- coding: utf-8 -*-
"""Process batches to generate clip embeddings."""

# Authors: Platon Lukyanenko, William La Cava

import time

import pandas as pd
import os
import shutil
import random

import torch
from tqdm import tqdm
import numpy as np


from torch.utils.data import Dataset, DataLoader
import h5py

import fire

from video_processing import Train_Transforms, Test_Transforms, pull_clips


class MyDataset(Dataset):
    """Dataset of video file paths and transforms."""

    def __init__(self, path_list, transforms):
        """Initialize dataset of video file paths.

        Args:
            path_list (list[str]): Video file paths.
            transforms (callable): Transform pipeline for clips.
        """
        self.path_list = path_list
        self.transforms = transforms

    def __len__(self):
        """Return number of files in the dataset."""
        return len(self.path_list)

    def __getitem__(self, idx):
        """Return stacked clips and file path for an index.

        Args:
            idx (int): Dataset index.

        Returns:
            tuple[torch.Tensor, str]: Clips tensor and file path.
        """
        return pull_clips(self.path_list[idx], self.transforms), self.path_list[idx]

def main(
    batch_num, out_dir="./out_loc", seed=0, train_transforms=False, parallel_count=8
):
    """Process a batch of trim folders into embedding files.

    Args:
        batch_num (int): Batch ID to process.
        out_dir (str): Output directory containing batch files and folders.
        seed (int): RNG seed for reproducibility.
        train_transforms (bool): Use training transforms if True.
        parallel_count (int): Number of parallel workers for the dataloader.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # load PanEcho
    model = torch.hub.load(
        "CarDS-Yale/PanEcho", "PanEcho", force_reload=False, backbone_only=True
    )
    model.to("cuda")
    model.eval()

    # okay, we have a batch number to work with
    print(batch_num, type(batch_num))

    batch_csv_path = os.path.join(out_dir, "Batches", str(batch_num) + ".csv")
    batch = pd.read_csv(batch_csv_path)

    # transform set

    transforms_to_use = Train_Transforms if train_transforms else Test_Transforms

    # okay, inside batch
    for trim_n in batch["batch_folder_names"]:
        # per folder, if done, skip
        if (trim_n + ".csv") in os.listdir(os.path.join(out_dir, "Complete")):
            print("already processed", (trim_n + ".csv"))
            continue

        # not done -> process
        tmp_df = pd.read_csv(os.path.join(out_dir, "Incomplete", trim_n + ".csv"))
        file_folder_path = tmp_df["Folder_Path"].values[0]  # path of trim folder

        print(file_folder_path)
        file_path_list = [
            os.path.join(file_folder_path, file)
            for file in os.listdir(file_folder_path)
        ]

        print("files found", len(file_path_list))

        start = time.time()
        dataset = MyDataset(file_path_list, transforms_to_use)
        dataloader = DataLoader(
            dataset,
            batch_size=parallel_count,
            num_workers=parallel_count,
            shuffle=False,
        )
        # dataloader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)

        clip_list = []
        returned_path_list = []
        embedding_list = []
        for i, (clips, file_path) in enumerate(
            tqdm(dataloader)
        ):  # 64 x 16 x 3 x 16 x 224 x 224
            clip_list.append(clips)
            returned_path_list.append(file_path)

            with torch.no_grad():
                embedding_list = embedding_list + [
                    model(k.to("cuda")).to("cpu") for k in clips
                ]

        clip_list = torch.vstack(clip_list)  # yields [N<256] * 16 * 3 * 16 * 224 * 224
        embedding_list = torch.stack(embedding_list, axis=0)  # N x 16 x 768

        embed_path = os.path.join(out_dir, "Embeddings", trim_n + "_embed.hdf5")
        with h5py.File(embed_path, "w") as f:  # f[keys[0]][keys_2[0]]['emb'][()]
            study_group = f.create_group(trim_n)
            for file_path, emb in zip(file_path_list, embedding_list):
                sub_group = study_group.create_group(file_path)
                sub_group.create_dataset("emb", data=emb)

        time_taken = (time.time() - start) / clip_list.shape[0]
        print("folder embedding time per file", time_taken)

        # move the trim_n.csv to 'complete'
        src_dr = os.path.join(out_dir, "Incomplete", trim_n + ".csv")
        dst_dr = os.path.join(out_dir, "Complete", trim_n + ".csv")
        shutil.move(src_dr, dst_dr)



if __name__ == "__main__":
    fire.Fire(main)
