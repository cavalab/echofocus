"""Dataset utilities for EchoFocus."""

import torch
import h5py
import os
import numpy as np
# %% data gathering
class EmbeddingDataset(torch.utils.data.Dataset):
    """Dataset that pulls clip embeddings from a study."""

    def __init__(self, embed_path, embedding_echo_id_list, 
        base_path = '/lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled',
        store_keychain=False,
        cache_embeddings=True
    ):
        """Create a dataset for loading precomputed clip embeddings.

        Args:
            embed_path (str): Folder containing ``*_trim_embed.hdf5`` files.
            embedding_echo_id_list (list[int]): Echo IDs to include.
            base_path (str): Base path used when reconstructing filenames.
            store_keychain (bool): Cache intermediate HDF5 path keys.
            cache_embeddings (bool): Cache loaded embeddings in memory.
        """
        self.embed_path = embed_path  # folder path
        self.embedding_echo_id_list = embedding_echo_id_list  # list of ints
        self.base_path=base_path
        self.keychain = []
        self.store_keychain = store_keychain
        self.cache_embeddings=cache_embeddings
        self.video_cache={}

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
            self.video_cache[echo_id] = study_clips
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

        return (
            torch.tensor(Embeddings),
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
    emb_ds = EmbeddingDataset(
        embed_path, embedding_echo_id_list, cache_embeddings=cache_embeddings
    )
    # todo: pass n workers to embeddingdataset
    if not preload:
        return emb_ds
    else:
        raise ValueError('preloading is deprecated. use cache_embeddings')
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
