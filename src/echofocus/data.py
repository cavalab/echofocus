"""Data loading and embedding-cache helpers."""

import hashlib
import json
import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .datasets import CustomDataset, custom_collate, get_dataset, get_video_dataset
from .video_processing import Test_Transforms, Train_Transforms


def panecho_cache_get(self, eid):
    if eid in self._panecho_cache:
        value = self._panecho_cache.pop(eid)
        self._panecho_cache[eid] = value
        return value
    return None


def panecho_cache_put(self, eid, value):
    if not self.cache_panecho_embeddings:
        return
    if self.max_panecho_cache_gb is None:
        self._panecho_cache[eid] = value
        return
    max_bytes = int(self.max_panecho_cache_gb * (1024 ** 3))
    size = value.numel() * value.element_size()
    if size > max_bytes:
        return
    while self._panecho_cache_bytes + size > max_bytes and len(self._panecho_cache) > 0:
        _, evicted = self._panecho_cache.popitem(last=False)
        self._panecho_cache_bytes -= evicted.numel() * evicted.element_size()
    self._panecho_cache[eid] = value
    self._panecho_cache_bytes += size


def panecho_cache_clear(self):
    self._panecho_cache.clear()
    self._panecho_cache_bytes = 0


def embedding_eids_from_path(self):
    """Return echo IDs available under the embedding path."""
    cache_index_path = os.path.join(self.embedding_path, "cache_index.json")
    if os.path.isfile(cache_index_path):
        with open(cache_index_path, "r") as f:
            cache_index = json.load(f)
        return [int(eid) for eid in cache_index.get("eids", [])]
    return [int(k.split("_")[0]) for k in os.listdir(self.embedding_path)]


def setup_data(self, input_norm_dict=None, use_train_transforms=True):
    """Prepare dataloaders and normalization metadata."""
    print('_setup_data...')
    print('label path:', self.label_path)
    csv_data = pd.read_csv(self.label_path)
    print('loaded', len(csv_data), 'labels from', self.label_path)
    print('dropping duplicates...')
    csv_data = csv_data.drop_duplicates()
    print('dropped duplicates, new length:', len(csv_data))
    if self.end_to_end and not self.use_hdf5_index:
        print('video_base_path:', self.video_base_path)
        candidate_eids = csv_data["eid"].astype(int).unique()
        print('candidate_eids:', candidate_eids)
        embedding_echo_id_list = [
            eid
            for eid in candidate_eids
            if os.path.isdir(
                os.path.join(
                    self.video_base_path,
                    self.video_subdir_format.format(echo_id=int(eid)),
                )
            )
        ]
    else:
        print('embed path:', self.embedding_path)
        embedding_echo_id_list = self._embedding_eids_from_path()

    print('Num echos in embedding folder:', len(embedding_echo_id_list))
    tmp = csv_data.copy()
    mask = tmp['eid'].isin(embedding_echo_id_list)
    tmp = tmp[mask]
    print('N echos after in_csv filter:', len(tmp))

    tmp = tmp.loc[tmp[self.task_labels].dropna(how='all').index]
    print('N Echos after excluding missing labels:', len(tmp))

    eid_keep_list = tmp['eid'].values
    study_embeddings = None
    if not self.end_to_end:
        study_embeddings = get_dataset(
            self.embedding_path,
            eid_keep_list,
            limit=self.sample_limit,
            parallel_processes=self.parallel_processes,
            cache_embeddings=self.cache_panecho_embeddings,
            max_cache_gb=self.max_panecho_cache_gb,
            batch_size=self.batch_size,
        )

    if self.preload_embeddings and not self.end_to_end:
        mask = tmp['eid'].isin(study_embeddings.keys())
        tmp = tmp[mask]
    new_csv_data = tmp

    new_csv_data.set_index('eid', inplace=True)
    pids = new_csv_data['pid'].astype(str).values
    unique_pids = np.unique(pids)

    if self.task == 'measure':
        tmp = new_csv_data['EF05'].values
        print(sum(tmp < 0), 'EF05 values below 0. setting to nan')
        tmp[tmp < 0] = np.nan
        new_csv_data['EF05'] = tmp

        tmp = new_csv_data['LM12'].values
        print(sum(tmp < 0), 'LM12 values below 0. setting to nan')
        tmp[tmp < 0] = np.nan
        new_csv_data['LM12'] = tmp

    tr, va, te = self.split
    print('train-val-test split:', tr, va, te)
    from torch.utils.data import random_split

    tr_count = int(np.ceil(len(unique_pids) * tr / (tr + va + te)))
    te_count = int(np.ceil((len(unique_pids) - tr_count) * te / (va + te)))
    v_count = len(unique_pids) - tr_count - te_count

    tr_ind, va_ind, te_ind = random_split(range(len(unique_pids)), [tr_count, v_count, te_count])

    tr_pid_list = unique_pids[tr_ind]
    va_pid_list = unique_pids[va_ind]
    te_pid_list = unique_pids[te_ind]

    train_df = new_csv_data[new_csv_data['pid'].astype(str).isin(tr_pid_list)]
    valid_df = new_csv_data[new_csv_data['pid'].astype(str).isin(va_pid_list)]
    test_df = new_csv_data[new_csv_data['pid'].astype(str).isin(te_pid_list)]

    print('Train_DF n=', len(train_df), ', pids:', train_df.pid.nunique())
    print('Valid_DF n=', len(valid_df), ', pids:', valid_df.pid.nunique())
    print('Test_DF n=', len(test_df), ', pids:', test_df.pid.nunique())

    if self.task == 'measure':
        from . import utils

        if input_norm_dict is None:
            print('no input_norm_dict loaded, generating from Train_DF')
            input_norm_dict = utils.get_norm_params(train_df, self.task_labels)
        train_df = utils.normalize_df(train_df, input_norm_dict)
        valid_df = utils.normalize_df(valid_df, input_norm_dict)
        test_df = utils.normalize_df(test_df, input_norm_dict)
        print('normalized labels')

    if self.end_to_end:
        test_embeddings = get_video_dataset(
            self.embedding_path,
            test_df.index.values,
            transforms=Test_Transforms,
            cache_clips=self.cache_video_tensors,
            num_clips=self.num_clips,
            clip_len=self.clip_len,
            base_path=self.video_base_path,
            use_hdf5_index=self.use_hdf5_index,
            video_subdir_format=self.video_subdir_format,
            max_videos_per_study=self.max_videos_per_study,
            max_cache_gb=self.max_video_cache_gb,
        )
        test_dataset = CustomDataset(test_df, test_embeddings, self.task_labels)
    else:
        test_dataset = CustomDataset(test_df, study_embeddings, self.task_labels)
    if self.sample_limit < len(test_dataset):
        print('subsampling test dataset')
        test_dataset = torch.utils.data.Subset(test_dataset, list(range(0, self.sample_limit)))
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=self.parallel_processes,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1 if self.parallel_processes else None,
    )

    if tr == 0:
        return None, None, test_dataloader, input_norm_dict

    if self.end_to_end:
        train_transform = Train_Transforms if use_train_transforms else Test_Transforms
        train_embeddings = get_video_dataset(
            self.embedding_path,
            train_df.index.values,
            transforms=train_transform,
            cache_clips=self.cache_video_tensors,
            num_clips=self.num_clips,
            clip_len=self.clip_len,
            base_path=self.video_base_path,
            use_hdf5_index=self.use_hdf5_index,
            video_subdir_format=self.video_subdir_format,
            max_videos_per_study=self.max_videos_per_study,
            max_cache_gb=self.max_video_cache_gb,
        )
        train_dataset = CustomDataset(train_df, train_embeddings, self.task_labels)
    else:
        train_dataset = CustomDataset(train_df, study_embeddings, self.task_labels)
    if self.sample_limit < len(train_dataset):
        print('subsampling train dataset')
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(0, self.sample_limit)))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.batch_size,
        collate_fn=custom_collate,
        num_workers=self.parallel_processes,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1 if self.parallel_processes else None,
    )

    if self.end_to_end:
        valid_embeddings = get_video_dataset(
            self.embedding_path,
            valid_df.index.values,
            transforms=Test_Transforms,
            cache_clips=self.cache_video_tensors,
            num_clips=self.num_clips,
            clip_len=self.clip_len,
            base_path=self.video_base_path,
            use_hdf5_index=self.use_hdf5_index,
            video_subdir_format=self.video_subdir_format,
            max_videos_per_study=self.max_videos_per_study,
            max_cache_gb=self.max_video_cache_gb,
        )
        valid_dataset = CustomDataset(valid_df, valid_embeddings, self.task_labels)
    else:
        valid_dataset = CustomDataset(valid_df, study_embeddings, self.task_labels)
    if self.sample_limit < len(valid_dataset):
        print('subsampling valid dataset')
        valid_dataset = torch.utils.data.Subset(valid_dataset, list(range(0, self.sample_limit)))
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=self.parallel_processes,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1 if self.parallel_processes else None,
    )

    return train_dataloader, valid_dataloader, test_dataloader, input_norm_dict


def cache_embeddings(
    self,
    cache_root="embed/cache",
    cache_tag=None,
    num_shards=512,
    compression="lzf",
    dtype="float16",
    use_train_transforms=False,
    overwrite=False,
    max_eids=None,
    amp=False,
    seed=None,
):
    """Cache PanEcho embeddings into sharded HDF5 files."""
    print("cache_embeddings: starting")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    print('label path:', self.label_path)
    csv_data = pd.read_csv(self.label_path, nrows=self.sample_limit)
    print('loaded', len(csv_data), 'labels from', self.label_path)
    print('dropping duplicates...')
    csv_data = csv_data.drop_duplicates()
    print('dropped duplicates')

    if not self.use_hdf5_index:
        print('video_base_path:', self.video_base_path)
        candidate_eids = csv_data["eid"].astype(int).unique()
        available_eids = [
            eid
            for eid in candidate_eids
            if os.path.isdir(
                os.path.join(
                    self.video_base_path,
                    self.video_subdir_format.format(echo_id=int(eid)),
                )
            )
        ]
    else:
        print('embed path:', self.embedding_path)
        available_eids = self._embedding_eids_from_path()

    print('Num echos available:', len(available_eids))
    tmp = csv_data.copy()
    mask = tmp['eid'].isin(available_eids)
    tmp = tmp[mask]
    print('N echos after in_csv filter:', len(tmp))
    tmp = tmp.loc[tmp[self.task_labels].dropna(how='all').index]
    print('N Echos after excluding missing labels:', len(tmp))
    eid_keep_list = tmp['eid'].astype(int).values
    if max_eids is not None:
        eid_keep_list = eid_keep_list[:max_eids]

    from .panecho import PanEchoBackbone

    device = "cuda" if torch.cuda.is_available() else "cpu"
    panecho = PanEchoBackbone(backbone_only=True, trainable=False).to(device)
    panecho.eval()

    def _hash_state_dict(state_dict):
        h = hashlib.sha1()
        for k in sorted(state_dict.keys()):
            h.update(k.encode("utf-8"))
            t = state_dict[k].detach().cpu().contiguous()
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(t.numpy().tobytes())
        return h.hexdigest()

    panecho_hash = _hash_state_dict(panecho.model.state_dict())
    transform_name = "Train_Transforms" if use_train_transforms else "Test_Transforms"
    cache_meta = {
        "version": 1,
        "panecho_hash": panecho_hash,
        "num_clips": int(self.num_clips),
        "clip_len": int(self.clip_len),
        "transform": transform_name,
        "num_shards": int(num_shards),
        "shard_format": "shard_{shard:05d}.h5",
        "dtype": dtype,
        "compression": compression,
        "video_subdir_format": self.video_subdir_format,
        "use_hdf5_index": bool(self.use_hdf5_index),
        "created_at": str(datetime.now()),
        "eids": [int(eid) for eid in eid_keep_list],
    }

    cache_key_payload = json.dumps(
        {
            "panecho_hash": panecho_hash,
            "num_clips": int(self.num_clips),
            "clip_len": int(self.clip_len),
            "transform": transform_name,
            "version": 1,
        },
        sort_keys=True,
    ).encode("utf-8")
    cache_key = hashlib.sha1(cache_key_payload).hexdigest()[:12]

    cache_dir_name = cache_tag if cache_tag else cache_key
    cache_dir = os.path.join(cache_root, cache_dir_name)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"cache_dir: {cache_dir}")
    cache_index_path = os.path.join(cache_dir, "cache_index.json")
    if not os.path.isfile(cache_index_path) or overwrite:
        with open(cache_index_path, "w") as f:
            json.dump(cache_meta, f, indent=2)
    else:
        print("cache_index.json already exists; reusing it.")

    transforms = Train_Transforms if use_train_transforms else Test_Transforms
    video_ds = get_video_dataset(
        self.embedding_path,
        eid_keep_list,
        transforms=transforms,
        cache_clips=False,
        num_clips=self.num_clips,
        clip_len=self.clip_len,
        base_path=self.video_base_path,
        use_hdf5_index=self.use_hdf5_index,
        video_subdir_format=self.video_subdir_format,
    )

    dtype_np = np.float16 if dtype == "float16" else np.float32
    use_amp = amp and torch.cuda.is_available()
    shard_map = {}
    for eid in eid_keep_list:
        shard_id = int(eid) % int(num_shards)
        shard_map.setdefault(shard_id, []).append(int(eid))

    for shard_id in sorted(shard_map.keys()):
        shard_path = os.path.join(cache_dir, f"shard_{shard_id:05d}.h5")
        with h5py.File(shard_path, "a") as f:
            for eid in tqdm(shard_map[shard_id], desc=f"Shard {shard_id:05d}"):
                if str(eid) in f and not overwrite and "emb" in f[str(eid)]:
                    continue
                clips, _ = video_ds[eid]
                clips = clips.to(device)
                embeddings = []
                with torch.no_grad():
                    for video_clips in clips:
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            video_emb = panecho(video_clips)
                        embeddings.append(video_emb.detach().cpu())
                emb = torch.stack(embeddings, dim=0).numpy().astype(dtype_np)
                grp = f.require_group(str(eid))
                if "emb" in grp:
                    del grp["emb"]
                grp.create_dataset("emb", data=emb, compression=compression)
    print("cache_embeddings: done")
