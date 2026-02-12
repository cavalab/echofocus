"""Dataset utilities for EchoFocus."""

import json
import torch
import h5py
import os
import numpy as np
from collections import OrderedDict
from video_processing import pull_clips
# %% data gathering
class EmbeddingDataset(torch.utils.data.Dataset):
    """Dataset that pulls clip embeddings from a study."""

    def __init__(self, embed_path, embedding_echo_id_list, 
        base_path = '/lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled',
        store_keychain=False,
        cache_embeddings=True,
        max_cache_gb=None,
    ):
        """Create a dataset for loading precomputed clip embeddings.

        Args:
            embed_path (str): Folder containing ``*_trim_embed.hdf5`` files.
            embedding_echo_id_list (list[int]): Echo IDs to include.
            base_path (str): Base path used when reconstructing filenames.
            store_keychain (bool): Cache intermediate HDF5 path keys.
            cache_embeddings (bool): Cache loaded embeddings in memory.
            max_cache_gb (float|None): Optional RAM cache cap in GB.
        """
        self.embed_path = embed_path  # folder path
        self.embedding_echo_id_list = embedding_echo_id_list  # list of ints
        self.base_path=base_path
        self.keychain = []
        self.store_keychain = store_keychain
        self.cache_embeddings=cache_embeddings
        self.max_cache_bytes = None if max_cache_gb is None else int(max_cache_gb * (1024 ** 3))
        self.video_cache = OrderedDict()
        self.cache_bytes = 0

    def _maybe_cache(self, echo_id, value):
        if not self.cache_embeddings:
            return
        if self.max_cache_bytes is not None:
            if torch.is_tensor(value):
                size = value.numel() * value.element_size()
            else:
                size = value.nbytes
            if size > self.max_cache_bytes:
                return
            while self.cache_bytes + size > self.max_cache_bytes and len(self.video_cache) > 0:
                _, evicted = self.video_cache.popitem(last=False)
                if torch.is_tensor(evicted):
                    evicted_size = evicted.numel() * evicted.element_size()
                else:
                    evicted_size = evicted.nbytes
                self.cache_bytes -= evicted_size
            self.video_cache[echo_id] = value
            self.cache_bytes += size
        else:
            self.video_cache[echo_id] = value

    def get_filenames(self, index):
        """Return filenames for an entry by dataset index.

        Args:
            index (int): Dataset index.

        Returns:
            tuple[numpy.ndarray, int]: Filenames array and echo ID.
        """
        return self.get_filenames_by_index(index)

    def get_filenames_by_index(self, index):
        """Return filenames for an entry by dataset index.

        Args:
            index (int): Dataset index.

        Returns:
            tuple[numpy.ndarray, int]: Filenames array and echo ID.
        """
        echo_id = self.embedding_echo_id_list[index]
        return self.get_filenames_by_echo_id(echo_id)

    def _get_subframe(self,f,echo_id):
        """Get the HDF5 group and base path for an echo ID.

        Args:
            f (h5py.File): Open HDF5 file handle.
            echo_id (int): Echo study identifier.

        Returns:
            tuple[h5py.Group, str]: Video group and resolved base path.
        """
        base_path = self.base_path
        study_key = f'{echo_id}_trim'
        f2 = f['lab-share']['Cardio-Mayourian-e2']['Public']
        if len(self.keychain)==0 or not self.store_keychain:
            i=0
            while study_key not in f2.keys() and i < 10:
                key = list(f2.keys())[0]
                f2 = f2[key] # go one level deeper
                base_path += '/'+key
                i += 1
                if self.store_keychain and key != study_key:
                    self.keychain.append(key)
            if self.store_keychain:
                print('EmbeddingDataset:: set self.keychain to',self.keychain)
        else:
            for key in self.keychain:
                f2 = f2[key] # go one level deeper
                base_path += '/'+key
        videos = f2[study_key]
        base_path += '/'+study_key
        return videos, base_path

    def get_filenames_by_echo_id(self, echo_id):
        """Return filenames for an entry by echo ID.

        Args:
            echo_id (int): Echo study identifier.

        Returns:
            tuple[numpy.ndarray, int]: Filenames array and echo ID.
        """
        embedding_path = os.path.join(
            self.embed_path, str(echo_id) + "_trim_embed.hdf5"
        )
        with h5py.File(embedding_path, "r") as f:
            videos,base_path = self._get_subframe(f, echo_id)
            # videos = f2[key].keys()
            study_filenames=[]
            for file in videos:
                study_filename = '/'.join([base_path,file])
                study_filenames.append(study_filename)
        study_filenames = np.array(study_filenames)
        return study_filenames, echo_id


    def get_by_index(self, index):
        """Return embeddings and echo ID by dataset index.

        Args:
            index (int): Dataset index.

        Returns:
            tuple[numpy.ndarray, int]: Clip embeddings and echo ID.
        """
        echo_id = self.embedding_echo_id_list[index]
        return self[echo_id]

    def __getitem__(self,echo_id):
        """Get all clip embeddings for an echo study.

        Args:
            echo_id (int): Echo study identifier.

        Returns:
            tuple[numpy.ndarray, int]: Clip embeddings and echo ID.
        """
        embedding_path = os.path.join(
            self.embed_path, str(echo_id) + "_trim_embed.hdf5"
        )
        if self.cache_embeddings and echo_id in self.video_cache:
            return  self.video_cache[echo_id], echo_id

        with h5py.File(embedding_path, "r") as f:
            videos,_ = self._get_subframe(f,echo_id)
            study_clips = []
            for k,v in videos.items():
                clip = v['emb'][()] # get clips
                study_clips.append(clip)
            study_clips = np.array(study_clips)
        if self.cache_embeddings:
            self._maybe_cache(echo_id, study_clips)
        return study_clips, echo_id

    def __len__(self):
        """Return dataset size."""
        return len(self.embedding_echo_id_list)

class ShardedEmbeddingDataset(torch.utils.data.Dataset):
    """Dataset that reads embeddings from sharded HDF5 cache files."""

    def __init__(self, cache_dir, embedding_echo_id_list, cache_embeddings=True):
        """Create a dataset for loading cached embeddings.

        Args:
            cache_dir (str): Cache directory containing ``cache_index.json``.
            embedding_echo_id_list (list[int]): Echo IDs to include.
            cache_embeddings (bool): Cache loaded embeddings in memory.
        """
        self.cache_dir = cache_dir
        self.embedding_echo_id_list = embedding_echo_id_list
        self.cache_embeddings = cache_embeddings
        self.video_cache = {}
        index_path = os.path.join(cache_dir, "cache_index.json")
        with open(index_path, "r") as f:
            self.cache_index = json.load(f)
        self.num_shards = int(self.cache_index["num_shards"])
        self.shard_format = self.cache_index["shard_format"]

    def _shard_path(self, echo_id):
        shard_id = int(echo_id) % self.num_shards
        return os.path.join(self.cache_dir, self.shard_format.format(shard=shard_id))

    def __getitem__(self, echo_id):
        """Get cached embeddings for an echo study."""
        if self.cache_embeddings and echo_id in self.video_cache:
            return self.video_cache[echo_id], echo_id

        shard_path = self._shard_path(echo_id)
        with h5py.File(shard_path, "r") as f:
            grp = f[str(int(echo_id))]
            study_clips = grp["emb"][()]

        if self.cache_embeddings:
            self.video_cache[echo_id] = study_clips
        return study_clips, echo_id

    def get_filenames_by_echo_id(self, echo_id):
        """Return empty filenames placeholder for cached embeddings."""
        return np.array([]), echo_id

    def __len__(self):
        """Return dataset size."""
        return len(self.embedding_echo_id_list)


class VideoClipDataset(torch.utils.data.Dataset):
    """Dataset that loads raw video clips for an echo study."""

    def __init__(
        self,
        embed_path,
        embedding_echo_id_list,
        transforms,
        base_path="/lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled",
        cache_clips=False,
        num_clips=16,
        clip_len=16,
        use_hdf5_index=False,
        video_subdir_format="{echo_id}_trim",
        max_videos_per_study=None,
        max_cache_gb=None,
    ):
        """Create a dataset for loading raw video clips.

        Args:
            embed_path (str): Folder containing ``*_trim_embed.hdf5`` files.
            embedding_echo_id_list (list[int]): Echo IDs to include.
            transforms (callable): Transform pipeline for clips.
            base_path (str): Base path used when reconstructing filenames.
            cache_clips (bool): Cache loaded clips in memory.
            num_clips (int): Number of clips to sample per video.
            clip_len (int): Frames per clip.
            use_hdf5_index (bool): Use embedding HDF5 files to locate video paths.
            video_subdir_format (str): Format for study folder under base_path.
            max_videos_per_study (int|None): Optional cap on videos per study.
            max_cache_gb (float|None): Optional RAM cache cap in GB.
        """
        self.embed_path = embed_path
        self.embedding_echo_id_list = embedding_echo_id_list
        self.transforms = transforms
        self.base_path = base_path
        self.cache_clips = cache_clips
        self.num_clips = num_clips
        self.clip_len = clip_len
        self.use_hdf5_index = use_hdf5_index
        self.video_subdir_format = video_subdir_format
        self.max_videos_per_study = max_videos_per_study
        self.keychain = []
        self.store_keychain = False #WGL: set to False to prevent parallel worker issue
        self.max_cache_bytes = None if max_cache_gb is None else int(max_cache_gb * (1024 ** 3))
        self.video_cache = OrderedDict()
        self.cache_bytes = 0

    def _maybe_cache(self, echo_id, value):
        if not self.cache_clips:
            return
        if self.max_cache_bytes is not None:
            size = value.numel() * value.element_size()
            if size > self.max_cache_bytes:
                return
            while self.cache_bytes + size > self.max_cache_bytes and len(self.video_cache) > 0:
                _, evicted = self.video_cache.popitem(last=False)
                evicted_size = evicted.numel() * evicted.element_size()
                self.cache_bytes -= evicted_size
            self.video_cache[echo_id] = value
            self.cache_bytes += size
        else:
            self.video_cache[echo_id] = value

    def _filter_video_files(self, files):
        exts = {
            ".avi",
            ".mp4",
            ".mov",
            ".mkv",
            ".mpg",
            ".mpeg",
            ".wmv",
            ".m4v",
            ".webm",
        }
        filtered = [f for f in files if os.path.splitext(f)[1].lower() in exts]
        return filtered if len(filtered) > 0 else files

    def _get_subframe(self, f, echo_id):
        """Get the HDF5 group and base path for an echo ID."""
        base_path = self.base_path
        study_key = f"{echo_id}_trim"
        f2 = f["lab-share"]["Cardio-Mayourian-e2"]["Public"]
        if len(self.keychain) == 0 or not self.store_keychain:
            i = 0
            while study_key not in f2.keys() and i < 10:
                key = list(f2.keys())[0]
                f2 = f2[key]  # go one level deeper
                base_path += "/" + key
                i += 1
                if self.store_keychain and key != study_key:
                    self.keychain.append(key)
        else:
            for key in self.keychain:
                f2 = f2[key]  # go one level deeper
                base_path += "/" + key
        videos = f2[study_key]
        base_path += "/" + study_key
        return videos, base_path

    def get_filenames_by_echo_id(self, echo_id):
        """Return filenames for an entry by echo ID."""
        if self.use_hdf5_index:
            embedding_path = os.path.join(
                self.embed_path, str(echo_id) + "_trim_embed.hdf5"
            )
            with h5py.File(embedding_path, "r") as f:
                videos, base_path = self._get_subframe(f, echo_id)
                study_filenames = []
                for file in videos:
                    study_filename = "/".join([base_path, file])
                    study_filenames.append(study_filename)
            study_filenames = self._filter_video_files(study_filenames)
            return np.array(study_filenames), echo_id

        study_dir = os.path.join(
            self.base_path,
            self.video_subdir_format.format(echo_id=int(echo_id)),
        )
        files = [
            os.path.join(study_dir, f)
            for f in os.listdir(study_dir)
            if os.path.isfile(os.path.join(study_dir, f))
        ]
        files = self._filter_video_files(sorted(files))
        return np.array(files), echo_id

    def __getitem__(self, echo_id):
        """Get all clip tensors for an echo study."""
        if self.cache_clips and echo_id in self.video_cache:
            return self.video_cache[echo_id], echo_id

        study_filenames, _ = self.get_filenames_by_echo_id(echo_id)
        if self.max_videos_per_study is not None:
            perm = np.random.permutation(len(study_filenames))
            study_filenames = study_filenames[perm][: int(self.max_videos_per_study)]
        study_clips = []
        for file_path in study_filenames:
            clip_tensor = pull_clips(
                file_path,
                self.transforms,
                num_clips=self.num_clips,
                clip_len=self.clip_len,
            )
            study_clips.append(clip_tensor)
        study_clips = torch.stack(study_clips, dim=0)

        if self.cache_clips:
            self._maybe_cache(echo_id, study_clips)
        return study_clips, echo_id

    def __len__(self):
        """Return dataset size."""
        return len(self.embedding_echo_id_list)

class CustomDataset(torch.utils.data.Dataset):
    """Dataset that pairs embeddings with label targets."""

    def __init__(self, dataframe, embeddings, task_labels,filenames=None):
        """Create a dataset of embeddings and labels.

        Args:
            dataframe (pandas.DataFrame): Label dataframe indexed by echo ID.
            embeddings (dict or Dataset): Embedding storage or dataset accessor.
            task_labels (list[str]): Target columns to predict.
            filenames (optional): Optional filenames mapping.
        """
        self.dataframe = dataframe
        self.embeddings = embeddings
        self.filenames = filenames
        self.task_labels = task_labels

    def __getitem__(self, index): 
        """Return embeddings, labels, and echo ID for an index.

        Args:
            index (int): Dataset index.

        Returns:
            tuple[torch.Tensor, torch.Tensor, int]: Embeddings, labels, and echo ID.
        """
        EID = self.dataframe.index[index]
        
        true_labels = self.dataframe.loc[EID][self.task_labels]
        # Embeddings = self.embeddings[EID]
        if isinstance(self.embeddings,dict):
            Embeddings = self.embeddings[EID]
        else:
            Embeddings,_ = self.embeddings[EID]  # [index]

        if torch.is_tensor(Embeddings):
            embeddings_tensor = Embeddings
        else:
            embeddings_tensor = torch.as_tensor(Embeddings)
        return (
            embeddings_tensor,
            torch.tensor(true_labels.values.astype(np.float32), dtype=torch.float32),
            EID,
        )

    def get_filenames(self, index):
        """Return filenames for the echo ID at the given index."""
        EID = self.dataframe.index[index]
        return self.embeddings.get_filenames_by_echo_id(EID)

    def __len__(self):
        """Return dataset size."""
        return self.dataframe.shape[0]

def emb_collate(batch):
    """Collate embedding batches for the embedding dataloader.

    Args:
        batch (list[tuple]): Sequence of (study_clips, echo_id).

    Returns:
        tuple[list, list]: List of clip arrays and list of echo IDs.
    """
    # group the clips sand the echo_ids
    study_clip_set = [k[0] for k in batch]
    echo_id_set = [k[1] for k in batch]
    return study_clip_set, echo_id_set

def get_dataset(
    embed_path,
    embedding_echo_id_list,
    limit=1e10,
    parallel_processes=1,
    preload=False,
    cache_embeddings=False,
    max_cache_gb=None,
    batch_size=1,
):
    """Create an embedding dataset or raise if preload is requested.

    Args:
        embed_path (str): Embedding folder path.
        embedding_echo_id_list (list[int]): Echo IDs to include.
        limit (int): Max number of studies to load when preloading (deprecated).
        parallel_processes (int): Number of workers (deprecated).
        preload (bool): Deprecated, raises if True.
        cache_embeddings (bool): Cache embeddings on first load.
        batch_size (int): Batch size used in deprecated preload path.

    Returns:
        EmbeddingDataset: Dataset that loads embeddings on demand.

    Raises:
        ValueError: If ``preload`` is True.
    """
    # returns all embeddings, which are in a folder per study, as a dict
    study_embeddings = {}
    study_filenames = {}
    cache_index_path = os.path.join(embed_path, "cache_index.json")
    if os.path.isfile(cache_index_path):
        emb_ds = ShardedEmbeddingDataset(
            embed_path, embedding_echo_id_list, cache_embeddings=cache_embeddings
        )
    else:
        emb_ds = EmbeddingDataset(
            embed_path,
            embedding_echo_id_list,
            cache_embeddings=cache_embeddings,
            max_cache_gb=max_cache_gb,
        )
    # todo: pass n workers to embeddingdataset
    if not preload:
        return emb_ds
    else:
        raise ValueError('preloading is deprecated. use cache_embeddings')


def get_video_dataset(
    embed_path,
    embedding_echo_id_list,
    transforms,
    cache_clips=False,
    num_clips=16,
    clip_len=16,
    base_path="/lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled",
    use_hdf5_index=False,
    video_subdir_format="{echo_id}_trim",
    max_videos_per_study=None,
    max_cache_gb=None,
):
    """Create a dataset that loads raw clips on demand."""
    return VideoClipDataset(
        embed_path,
        embedding_echo_id_list,
        transforms=transforms,
        cache_clips=cache_clips,
        num_clips=num_clips,
        clip_len=clip_len,
        base_path=base_path,
        use_hdf5_index=use_hdf5_index,
        video_subdir_format=video_subdir_format,
        max_videos_per_study=max_videos_per_study,
        max_cache_gb=max_cache_gb,
    )
    # print('Starting Embedding Pull')
    # start = time.time()
    # parallel_processes = 8
    # if parallel_processes > 1:
    #     emb_dl = torch.utils.data.DataLoader(
    #         emb_ds,
    #         batch_size=batch_size,
    #         collate_fn=emb_collate,
    #         shuffle=False,
    #         num_workers=parallel_processes,
    #     )
    # else:
    #     emb_dl = torch.utils.data.DataLoader(
    #         emb_ds, batch_size=batch_size, collate_fn=emb_collate, shuffle=False
    #     )

    # study_count = 0
    # for i,(study_clip_set, echo_id_set) in tqdm(enumerate(emb_dl),total=len(emb_dl.dataset)):
    #     for study_clip, echo_id in zip(study_clip_set, echo_id_set):
    #         study_embeddings[echo_id] = study_clip
    #         study_filenames[echo_id] = emb_dl.dataset.get_filenames(i)
    #         study_count += 1

    #     if study_count >= limit:
    #         print("study embedding limit reached:", limit)
    #         break

    # print('Dataset entries: ', study_count)
    # print(f'Done. Time: {time.time() - start:.2f} seconds')
    # print('Total embeddings (studies) pulled: ',len(study_embeddings))
    # return study_embeddings 

def custom_collate(batch):
    """Collate a batch from :class:`CustomDataset`.

    This implementation returns the first (and only) element because the
    dataloader uses ``batch_size=1``.

    Args:
        batch (list[tuple]): Sequence of (embeddings, labels, echo_id).

    Returns:
        tuple: (embeddings, labels, echo_id) for the first item.
    """
    # input: a list of tuples (n)
    # each tuple contains:
    # clips (4 x 3 x 16 x 224 x x224)
    # correct_lvef_tensor (4)
    # pid (list, 4)
    # output: (nx4) x 3 x 16 x 224 x 224 echo clips, nx4 lvef tensors, nx4-length-list

    # clip_stacked = torch.vstack([k[0] for k in batch]) #  (nx4) x 3 x 16x 224 x 224
    # correct_lvef_stacked = torch.vstack([k[1] for k in batch]) # (nx4)
    # pid_stacked = []
    # for k in batch:
    #     pid_stacked += k[2]

    # return clip_stacked, correct_lvef_stacked, pid_stacked
    # import ipdb
    # ipdb.set_trace()
    return batch[0][0], batch[0][1], batch[0][2]
    # return [
    #     (batch[i][0], batch[i][1], batch[i][2])
    # ]
    # return batch
