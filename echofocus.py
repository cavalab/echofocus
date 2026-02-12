# -*- coding: utf-8 -*-
"""Main entry point for training and evaluating EchoFocus models.

Authors: Platon Lukyanenko, William La Cava
"""

# -1. imports.
import fire
import pandas as pd
import os
import json
import csv
import h5py

import torch
import time
from datetime import datetime
import sys
import torch.profiler
import threading
import subprocess
import torch.multiprocessing as mp
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, average_precision_score, median_absolute_error, r2_score

# import cv2
import numpy as np
# from torchvision import tv_tensors
# from torchvision.transforms import resize, center_crop

# from torchvision.transforms import v2


from tqdm import tqdm
import uuid

import utils
from datasets import CustomDataset, get_dataset, get_video_dataset, custom_collate
from models import CustomTransformer, EchoFocusEndToEnd
from video_processing import Train_Transforms, Test_Transforms


class EchoFocus:
    """Train, evaluate, and explain EchoFocus models."""

    @utils.initializer  # this decorator automatically sets arguments to class attributes.
    def __init__(
        self,
        model_name=None,
        dataset=None,
        task='measure',
        seed=0,
        batch_number=128, # number of batches processed before updating
        batch_size=1,
        total_epochs=10,
        epoch_early_stop=9999,
        learning_rate=0.0001,  # default to 1e-4
        encoder_depth=0,
        clip_dropout=0.,
        tf_combine='avg',
        debug_echo_folder=False,
        laptop_debug=False,
        test_only=False,
        parallel_processes=1,
        sample_limit=1e10,
        preload_embeddings=False,
        run_id=None,
        config='config.json',
        cache_video_tensors=False,
        cache_panecho_embeddings=False,
        end_to_end=True,
        panecho_trainable=True,
        transformer_trainable=True,
        load_transformer_path=None,
        load_panecho_path=None,
        load_strict=False,
        num_clips=16,
        clip_len=16,
        use_hdf5_index=False,
        label_path=None,
        embedding_path=None,
        video_base_path="/lab-share/Cardio-Mayourian-e2/Public/Echo_Pulled",
        video_subdir_format="{echo_id}_trim",
        max_videos_per_study=None,
        max_video_cache_gb=250,
        max_panecho_cache_gb=None,
        smoke_train=False,
        smoke_num_steps=2,
        debug_mem=False,
        amp=False,
        checkpoint_panecho=False,
        profile=False,
        profile_steps=20,
        profile_dir="profiles",
        timing_every=None,
        profile_summary=False,
        gpu_monitor=False,
        gpu_monitor_interval=10,
        ram_monitor=False,
        ram_monitor_interval=10,
        sharing_strategy="file_descriptor",
    ):
        """Initialize training/evaluation state and load config.

        Args:
            model_name (str|None): Name for the model run directory.
            dataset (str|None): Dataset key in the config file.
            task (str): Task key in the config file.
            seed (int): RNG seed for reproducibility.
            batch_number (int): Gradient accumulation steps.
            batch_size (int): Batch size (only 1 supported).
            total_epochs (int): Max epochs to train; -1 for eval-only.
            epoch_early_stop (int): Early stopping patience in epochs.
            learning_rate (float): Optimizer learning rate.
            encoder_depth (int): Number of transformer encoder layers.
            clip_dropout (float): Dropout probability for clip embeddings.
            tf_combine (str): Pooling method for transformer output.
            debug_echo_folder (bool): Debug flag for local echo folder.
            laptop_debug (bool): Debug flag for local laptop use.
            test_only (bool): If True, run evaluation only.
            parallel_processes (int): Number of dataloader workers.
            sample_limit (int): Limit number of samples.
            run_id (str|None): Optional run ID for reproducibility.
            config (str): Path to config JSON file.
            cache_video_tensors (bool): Cache raw video clips in memory (end-to-end only).
            cache_panecho_embeddings (bool): Cache PanEcho embeddings (end-to-end frozen) or precomputed embeddings (non end-to-end).
            end_to_end (bool): If True, train end-to-end with PanEcho backbone.
            panecho_trainable (bool): If True, fine-tune the PanEcho backbone.
            transformer_trainable (bool): If True, train the study-level transformer.
            load_transformer_path (str|None): Optional path to load transformer weights.
            load_panecho_path (str|None): Optional path to load PanEcho weights.
            load_strict (bool): If True, enforce strict loading for submodules.
            num_clips (int): Number of clips sampled per video.
            clip_len (int): Frames per clip.
            use_hdf5_index (bool): Use embedding HDF5s to locate video paths.
            video_base_path (str): Base path for raw study folders.
            video_subdir_format (str): Format for study folder under base path.
            max_videos_per_study (int|None): Optional cap on videos per study.
            max_video_cache_gb (float|None): Optional RAM cache cap for video tensors.
            max_panecho_cache_gb (float|None): Optional RAM cache cap for PanEcho embeddings.
            smoke_train (bool): If True, run a minimal smoke-training pass.
            smoke_num_steps (int): Number of batches per epoch for smoke training.
            debug_mem (bool): If True, print CUDA memory stats for first batch each epoch.
            amp (bool): If True, use autocast + GradScaler mixed precision.
            checkpoint_panecho (bool): If True, checkpoint PanEcho forward to save memory.
            profile (bool): If True, run PyTorch profiler for a short window.
            profile_steps (int): Number of steps to profile.
            profile_dir (str): Output directory for profiler traces.
            timing_every (int): Print timing stats every N batches.
            profile_summary (bool): If True, print a summary table after profiling.
            gpu_monitor (bool): If True, log GPU utilization periodically.
            gpu_monitor_interval (int): Seconds between GPU utilization logs.
            ram_monitor (bool): If True, log process RAM usage periodically.
            ram_monitor_interval (int): Seconds between RAM usage logs.
            sharing_strategy (str): torch.multiprocessing sharing strategy ("file_descriptor" or "file_system").
        """
        self.time = time.time()
        self.datetime = str(datetime.now()).replace(" ", "_")
        if run_id:
            self.run_id = run_id
        else:
            self.run_id = f"{self.datetime}_{uuid.uuid4()}"

        self.cache_video_tensors = cache_video_tensors
        self.cache_panecho_embeddings = cache_panecho_embeddings
        self.max_video_cache_gb = max_video_cache_gb
        self.max_panecho_cache_gb = max_panecho_cache_gb
        self._panecho_cache = OrderedDict()
        self._panecho_cache_bytes = 0

        if "TORCH_SHM_DIR" not in os.environ:
            tmp_base = os.environ.get("TMPDIR") or os.environ.get("TMP") or os.environ.get("TEMP")
            if tmp_base:
                os.environ["TORCH_SHM_DIR"] = os.path.join(tmp_base, "torch-shm")

        if self.sharing_strategy in ("file_descriptor", "file_system"):
            mp.set_sharing_strategy(self.sharing_strategy)
        else:
            print(f"WARNING: unknown sharing_strategy={self.sharing_strategy}; using file_descriptor")
            mp.set_sharing_strategy("file_descriptor")

        try:
            shm_stats = subprocess.check_output(["df", "-h", "/dev/shm"], text=True).strip().splitlines()
            shm_line = shm_stats[-1] if shm_stats else ""
        except Exception:
            shm_line = "unavailable"
        print(
            "preflight:",
            f"sharing_strategy={mp.get_sharing_strategy()}",
            f"TORCH_SHM_DIR={os.environ.get('TORCH_SHM_DIR', '')}",
            f"TMPDIR={os.environ.get('TMPDIR', '')}",
            f"/dev/shm={shm_line}",
        )
        print(
            "cache:",
            f"video_tensors={self.cache_video_tensors}",
            f"panecho_embeddings={self.cache_panecho_embeddings}",
            f"max_video_cache_gb={self.max_video_cache_gb}",
            f"max_panecho_cache_gb={self.max_panecho_cache_gb}",
            f"num_clips={self.num_clips}",
            f"clip_len={self.clip_len}",
            f"max_videos_per_study={self.max_videos_per_study}",
        )

        assert batch_size==1, "only batch_size=1 currently supported"
        print('main')
        args = {**locals()}
        # input is paired dict of strings named args
        start_time = time.time()

        print("random seed", seed, "\n")
        print("batch_number", batch_number, "\n")

        if total_epochs == -1:
            print("epoch lim missing. evaluating model")
        print("total_epochs", total_epochs, "\n")

        if epoch_early_stop == 9999:
            print("no early stop. defaulting to 10k epochs")
        print("epoch_early_stop", epoch_early_stop, "\n")

        print("learning_rate", learning_rate, "\n")

        self._cli_overrides = set()
        for arg in sys.argv[1:]:
            if not arg.startswith("--"):
                continue
            key = arg[2:]
            if "=" in key:
                key = key.split("=", 1)[0]
            self._cli_overrides.add(key)


        # 1. Check cuda
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i).name)
            _ = torch.tensor(3).to('cuda:'+str(i)) # test CUDA device (sometimes crashes)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(0)          # or your desired device / local_rank
            torch.cuda.init()                 # explicitly initialize CUDA context
            _ = torch.empty(1, device="cuda") # tiny warmup alloc (optional but common)
        else:
            raise ValueError('No CUDA. Exiting.')
        
                
        # 2. Set random seeds 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = False  # faster kernels, non-deterministic
        torch.backends.cudnn.benchmark = True       # auto-tune cuDNN for fixed input sizes
         
        # set model name 
        if not model_name:
            model_name = f'{task}_{self.run_id}'
        self.model_path = os.path.join('./trained_models', model_name)
        os.makedirs(self.model_path,exist_ok=True)

        self._load_config()
        self._set_loss()

    def _start_gpu_monitor(self):
        if not self.gpu_monitor:
            return None, None
        stop_event = threading.Event()

        def _worker():
            while not stop_event.is_set():
                try:
                    out = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,utilization.memory",
                            "--format=csv,noheader,nounits",
                        ],
                        stderr=subprocess.DEVNULL,
                        text=True,
                    ).strip()
                    if out:
                        parts = [p.strip() for p in out.split(",")]
                        if len(parts) >= 2:
                            self._gpu_status = f"{parts[0]}%/{parts[1]}%"
                        else:
                            self._gpu_status = out
                except Exception:
                    pass
                stop_event.wait(self.gpu_monitor_interval)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return thread, stop_event

    def _start_ram_monitor(self):
        if not self.ram_monitor:
            return None, None
        stop_event = threading.Event()

        def _read_rss_mb():
            try:
                import psutil
                proc = psutil.Process(os.getpid())
                return proc.memory_info().rss / (1024 ** 2)
            except Exception:
                try:
                    with open(f"/proc/{os.getpid()}/statm", "r") as f:
                        parts = f.read().strip().split()
                    if len(parts) >= 2:
                        rss_pages = int(parts[1])
                        return rss_pages * (os.sysconf("SC_PAGE_SIZE") / (1024 ** 2))
                except Exception:
                    return None
            return None

        def _worker():
            while not stop_event.is_set():
                rss_mb = _read_rss_mb()
                if rss_mb is not None:
                    self._ram_status = f"{rss_mb/1024:.2f}GB"
                stop_event.wait(self.ram_monitor_interval)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return thread, stop_event

    def _load_submodule_weights(self, module, path, prefix=None):
        if path is None:
            return
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        if prefix:
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        missing, unexpected = module.load_state_dict(state, strict=self.load_strict)
        if missing or unexpected:
            print(
                f"WARNING: load from {path} missing={len(missing)} unexpected={len(unexpected)}"
            )

    def _panecho_cache_get(self, eid):
        if eid in self._panecho_cache:
            value = self._panecho_cache.pop(eid)
            self._panecho_cache[eid] = value
            return value
        return None

    def _panecho_cache_put(self, eid, value):
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

    def _panecho_cache_clear(self):
        self._panecho_cache.clear()
        self._panecho_cache_bytes = 0

    def _embedding_eids_from_path(self):
        """Return echo IDs available under the embedding path."""
        cache_index_path = os.path.join(self.embedding_path, "cache_index.json")
        if os.path.isfile(cache_index_path):
            with open(cache_index_path, "r") as f:
                cache_index = json.load(f)
            return [int(eid) for eid in cache_index.get("eids", [])]
        return [int(k.split("_")[0]) for k in os.listdir(self.embedding_path)]

    def _set_trainable_flags(self):
        """Apply trainable flags to model submodules."""
        if not hasattr(self, "model") or self.model is None:
            return
        if hasattr(self.model, "panecho"):
            for param in self.model.panecho.parameters():
                param.requires_grad = bool(self.panecho_trainable)
        if hasattr(self.model, "transformer"):
            for param in self.model.transformer.parameters():
                param.requires_grad = bool(self.transformer_trainable)

    def _get_last_epoch(self):
        """Return last trained epoch from checkpoint, or 0 if missing."""
        last_ckpt = os.path.join(self.model_path, "last_checkpoint.pt")
        if not os.path.isfile(last_ckpt):
            return 0
        try:
            ckpt = torch.load(last_ckpt, map_location="cpu")
            if "perf_log" in ckpt and len(ckpt["perf_log"]) > 0:
                return ckpt["perf_log"][-1][0]
        except Exception:
            pass
        return 0

    def _load_config(self):
        """Load dataset/task config and set instance attributes.

        Raises:
            ValueError: If dataset is not defined in the config.
            AssertionError: If task is not defined in the config.
        """
        with open(self.config,'r') as f:
            data = json.load(f)

        assert self.task in data['task'].keys(), f'task must be one of: {data["task"].keys()}; got "{self.task}"' 

        if self.dataset not in data['dataset'].keys():
            raise ValueError(f'dataset must be one of: {list(data["dataset"].keys())}; got \"{self.dataset}\"')
        for k,v in data['task'][self.task].items():
            if k in self._cli_overrides:
                continue
            if hasattr(self, k) and getattr(self, k) != v:
                print(f'WARNING: overriding init arg {k}={getattr(self, k)} with config value {v}')
            setattr(self,k,v)
        for k,v in data['dataset'][self.dataset].items():
            if k in self._cli_overrides:
                continue
            if hasattr(self, k) and getattr(self, k) != v:
                print(f'WARNING: overriding init arg {k}={getattr(self, k)} with config value {v}')
            setattr(self,k,v)

    def save(self):
        """Save run parameters to ``cfg.json`` in the run directory."""
        self.time = time.time() - self.time
        save_name = f"{self.save_dir}/{self.run_id}/cfg.json"
        with open(save_name, "w") as of:
            payload = {
                k: v
                for k, v in vars(self).items()
                if any(isinstance(v, t) for t in [bool, int, float, str, dict, list, tuple])
            }
            print("payload:", json.dumps(payload, indent=2))
            json.dump(payload, of, indent=4)

    def _setup_data(self, input_norm_dict=None, use_train_transforms=True):
        """Prepare dataloaders and normalization metadata.

        Args:
            input_norm_dict (dict|None): Existing normalization parameters.

        Returns:
            tuple: (train_dataloader, valid_dataloader, test_dataloader, input_norm_dict)
        """
        print('_setup_data...')
        print('label path:',self.label_path)
        csv_data = pd.read_csv(self.label_path) # pull labels from local path
        print('loaded',len(csv_data),'labels from',self.label_path)
        print('dropping duplicates...')
        csv_data = csv_data.drop_duplicates() # I don't know why there are duplicates, but there are...
        print('dropped duplicates, new length:',len(csv_data))
        # if self.sample_limit < len(csv_data):
        #     print('sampling csv_data')
        if self.end_to_end and not self.use_hdf5_index:
            print('video_base_path:',self.video_base_path)
            candidate_eids = csv_data["eid"].astype(int).unique()
            print('candidate_eids:',candidate_eids)
            Embedding_EchoID_List = [
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
            print('embed path:',self.embedding_path)
            Embedding_EchoID_List = self._embedding_eids_from_path()

        print('Num echos in embedding folder:',len(Embedding_EchoID_List))
        # 3.2 limit label df rows to those
        tmp = csv_data.copy()
        mask = tmp['eid'].isin(Embedding_EchoID_List)
        tmp = tmp[mask]
        print('N echos after in_csv filter:',len(tmp))
        
        # 3.3 also limit label df rows to those we have any regression measure for
        tmp = tmp.loc[tmp[self.task_labels].dropna(how='all').index]
        # tmp = tmp[mask]    
        print('N Echos after excluding missing labels:',len(tmp))
        
        # 3.4 that sets which embeddings we load
        eid_keep_list =  tmp['eid'].values
        
        # study_embeddings, study_filenames = get_dataset(embedding_path, eid_keep_list, limit=sample_limit, parallel_processes=parallel_processes)
        study_embeddings = None
        if not self.end_to_end:
            study_embeddings = get_dataset(
                self.embedding_path,
                eid_keep_list,
                limit=self.sample_limit,
                parallel_processes=self.parallel_processes,
                cache_embeddings=self.cache_panecho_embeddings,
                max_cache_gb=self.max_panecho_cache_gb,
                batch_size=self.batch_size
            )
        # print('Total videos included: ',sum([study_embeddings[key].shape[0] for key in study_embeddings.keys()]))
        # so study_embeddings is a dict of M x 16 x 768, indexed by echo ID (EID)
        
        #because of laptop_debug we don't always keep all the eids. limit the dataframe to what we pulled
        if self.preload_embeddings and not self.end_to_end:
            mask = tmp['eid'].isin(study_embeddings.keys())
            tmp = tmp[mask]
        eid_keep_list =  tmp['eid'].values
        new_csv_data=tmp
        
        # Clips is 200k x 16 x 728
        # PIDs is 200k
        # video_names is 200k
        
        new_csv_data.set_index('eid',inplace=True)
        PIDs = new_csv_data['pid'].astype(str).values
        Unique_PIDs = np.unique(PIDs)

        if self.task == 'measure': 
            # adjust for plausibilitiy
            tmp = new_csv_data['EF05'].values
            print(sum(tmp<0),'EF05 values below 0. setting to nan')
            tmp[tmp<0] = np.nan
            new_csv_data['EF05'] = tmp
            
            tmp = new_csv_data['LM12'].values
            print(sum(tmp<0),'LM12 values below 0. setting to nan')
            tmp[tmp<0] = np.nan
            new_csv_data['LM12'] = tmp
        
        # 4. now we have multiple videos per PID ... split data by PID
        # Tr = 64
        # Va = 16
        # Te = 20
        Tr, Va, Te = self.split
        print('train-val-test split:',Tr,Va,Te)
        from torch.utils.data import random_split
        Tr_Count = int(np.ceil(len(Unique_PIDs) * Tr / (Tr + Va + Te)))
        Te_Count = int(np.ceil( (len(Unique_PIDs) - Tr_Count) * Te / (Va + Te)))
        V_Count = len(Unique_PIDs) - Tr_Count - Te_Count
        
        Tr_Ind, Va_Ind, Te_Ind = random_split(range(len(Unique_PIDs)), [Tr_Count, V_Count, Te_Count])
        
        Tr_PID_list = Unique_PIDs[Tr_Ind]
        Va_PID_list = Unique_PIDs[Va_Ind]
        Te_PID_list = Unique_PIDs[Te_Ind]
        # WGL: save test set
        # test_csv_data=new_csv_data.loc[new_csv_data['pid'].astype(str).isin(Te_PID_list)] 
        # test_csv_data.to_csv(f'{self.dataset.lower()}_echo_measurements_test.csv')

        Train_DF = new_csv_data[new_csv_data['pid'].astype(str).isin(Tr_PID_list)]
        Valid_DF = new_csv_data[new_csv_data['pid'].astype(str).isin(Va_PID_list)]
        Test_DF  = new_csv_data[new_csv_data['pid'].astype(str).isin(Te_PID_list)]

        print('Train_DF n=',len(Train_DF),', pids:',Train_DF.pid.nunique())
        print('Valid_DF n=',len(Valid_DF),', pids:',Valid_DF.pid.nunique())
        print('Test_DF n=',len(Test_DF),', pids:',Test_DF.pid.nunique())


        # Get normalization parameters, normalize datasets
        # if (('input_norm_dict' not in locals()) or (input_norm_dict is None)): # if didn't get or never had
        if self.task=='measure':
            if input_norm_dict is None:
                print('no input_norm_dict loaded, generating from Train_DF')
                input_norm_dict = utils.get_norm_params(Train_DF, self.task_labels)
            Train_DF = utils.normalize_df(Train_DF,input_norm_dict)
            Valid_DF = utils.normalize_df(Valid_DF,input_norm_dict)
            Test_DF = utils.normalize_df(Test_DF,input_norm_dict)
            print('normalized labels')

        if self.end_to_end:
            test_embeddings = get_video_dataset(
                self.embedding_path,
                Test_DF.index.values,
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
            test_dataset = CustomDataset(Test_DF, test_embeddings, self.task_labels)
        else:
            test_dataset = CustomDataset(Test_DF, study_embeddings, self.task_labels) #, study_filenames)
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

        if (Tr == 0):
            return None, None, test_dataloader, input_norm_dict
        else:
            # weights = np.ones(len(Train_DF))
            if self.end_to_end:
                train_transform = Train_Transforms if use_train_transforms else Test_Transforms
                train_embeddings = get_video_dataset(
                    self.embedding_path,
                    Train_DF.index.values,
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
                train_dataset = CustomDataset(Train_DF, train_embeddings, self.task_labels)
            else:
                train_dataset = CustomDataset(Train_DF, study_embeddings, self.task_labels)  # , study_filenames)
            if self.sample_limit < len(train_dataset):
                print('subsampling train dataset')
                train_dataset = torch.utils.data.Subset(train_dataset, list(range(0, self.sample_limit)))
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                collate_fn=custom_collate,
                # sampler=torch.utils.data.WeightedRandomSampler(
                #     weights, len(weights), replacement=True
                # ),
                num_workers=self.parallel_processes,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=1 if self.parallel_processes else None,
            )
            
            if self.end_to_end:
                valid_embeddings = get_video_dataset(
                    self.embedding_path,
                    Valid_DF.index.values,
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
                valid_dataset = CustomDataset(Valid_DF, valid_embeddings, self.task_labels)
            else:
                valid_dataset = CustomDataset(Valid_DF, study_embeddings, self.task_labels) #, study_filenames)
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
        """Cache PanEcho embeddings into sharded HDF5 files.

        Args:
            cache_root (str): Root directory for cached embeddings.
            cache_tag (str|None): Optional cache directory name override.
            num_shards (int): Number of HDF5 shard files to write.
            compression (str|None): HDF5 compression ("lzf", "gzip", or None).
            dtype (str): Numpy dtype for stored embeddings ("float16" or "float32").
            use_train_transforms (bool): If True, use Train_Transforms.
            overwrite (bool): If True, overwrite existing cached embeddings.
            max_eids (int|None): Limit number of echo IDs to cache.
            amp (bool): If True, use autocast for embedding generation.
            seed (int|None): Optional RNG seed for deterministic clip sampling.
        """
        print("cache_embeddings: starting")
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        print('label path:',self.label_path)
        csv_data = pd.read_csv(self.label_path, nrows=self.sample_limit)
        print('loaded',len(csv_data),'labels from',self.label_path)
        print('dropping duplicates...')
        csv_data = csv_data.drop_duplicates()
        print('dropped duplicates')

        if not self.use_hdf5_index:
            print('video_base_path:',self.video_base_path)
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
            print('embed path:',self.embedding_path)
            available_eids = self._embedding_eids_from_path()

        print('Num echos available:',len(available_eids))
        tmp = csv_data.copy()
        mask = tmp['eid'].isin(available_eids)
        tmp = tmp[mask]
        print('N echos after in_csv filter:',len(tmp))
        tmp = tmp.loc[tmp[self.task_labels].dropna(how='all').index]
        print('N Echos after excluding missing labels:',len(tmp))
        eid_keep_list = tmp['eid'].astype(int).values
        if max_eids is not None:
            eid_keep_list = eid_keep_list[:max_eids]

        from panecho import PanEchoBackbone
        device = "cuda" if torch.cuda.is_available() else "cpu"
        panecho = PanEchoBackbone(backbone_only=True, trainable=False).to(device)
        panecho.eval()

        def _hash_state_dict(state_dict):
            import hashlib
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

        import hashlib
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
                    if str(eid) in f and not overwrite:
                        if "emb" in f[str(eid)]:
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

    # def _normalize_data(self):

    def _setup_model(self):
        """Initialize model, optimizer, scheduler, and load checkpoints.

        Returns:
            tuple: (model, current_epoch, best_epoch, best_loss, input_norm_dict)
        """
        print('_setup_model...')
        # 5. Set up folders and save training args
        self.last_checkpoint_path = os.path.join(self.model_path, 'last_checkpoint.pt') 
        self.best_checkpoint_path = os.path.join(self.model_path, 'best_checkpoint.pt') 
        self.log_path = os.path.join(self.model_path, 'train_losses.csv')
        
        # csvpath = os.path.join(self.model_path, 'train_args.csv')
        # with open(csvpath,'w',newline='') as f:
        #     headers = [key for key in args.keys()]
        #     values = [args[key] for key in args.keys()]
        #     writer = csv.writer(f,delimiter=',')
        #     writer.writerow(headers)
        #     writer.writerow(values)

        # if there is a trained model we are loading, make sure the training
        # arguments related to CustomTransformer match what was used. 
        # if they don't, override them and warn the user.
        train_args_path = os.path.join(self.model_path,'train_args.csv')
        if os.path.exists(train_args_path):
            train_args = pd.read_csv(train_args_path).to_dict(orient='records')[0]
            for k in ['encoder_depth','task_labels','clip_dropout']:
                if k in train_args and train_args[k] != getattr(self,k):
                    print(f'WARNING: using {k}={train_args[k]}, loaded from {train_args_path}')
                    setattr(self,k,train_args[k])

        # 6. Pull model if it already exists
        if self.end_to_end:
            self.model = EchoFocusEndToEnd(
                input_size=768,
                encoder_dim=768,
                n_encoder_layers=self.encoder_depth,
                output_size=len(self.task_labels),
                clip_dropout=self.clip_dropout,
                tf_combine="avg",
                panecho_trainable=self.panecho_trainable,
                debug_mem=self.debug_mem,
                checkpoint_panecho=self.checkpoint_panecho,
            )
        else:
            self.model = CustomTransformer(
                input_size=768,
                encoder_dim=768,
                n_encoder_layers=self.encoder_depth,
                output_size=len(self.task_labels),
                clip_dropout=self.clip_dropout,
                tf_combine="avg",
            )
        self._set_trainable_flags()
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(
            "model parameters:",
            f"total={total_params:,}",
            f"trainable={trainable_params:,}",
        )
    
        if (torch.cuda.is_available()):
            self.model = self.model.to('cuda')
        elif self.amp:
            print("WARNING: amp=True requested but CUDA is not available; running in full precision on CPU.")

        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.learning_rate, weight_decay = 0.01)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        # add the scheduler
        patience = 3
        lr_factor = 0.5
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, factor=lr_factor)

        self.perf_log = []
        input_norm_dict=None
        if (os.path.isfile(self.last_checkpoint_path)):
            print('loading lastcheckpoint')
            self.model, self.optimizer, self.scheduler, self.perf_log, input_norm_dict = (
                load_model_and_random_state(
                    self.last_checkpoint_path,
                    self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
            )
            current_epoch = self.perf_log[-1][0] 
            tmp = np.array(self.perf_log)
            best_epoch = tmp[np.argmin(tmp[:,2]),0]
            best_loss  = tmp[np.argmin(tmp[:,2]),2]
        
        # Otherwise initialize model
        else:
            current_epoch = 0
            best_epoch = 0
            best_loss = 1e10

        # override submodule weights after any full checkpoint load
        if self.end_to_end:
            if self.load_panecho_path:
                self._load_submodule_weights(self.model.panecho, self.load_panecho_path, prefix="panecho.")
            if self.load_transformer_path:
                self._load_submodule_weights(self.model.transformer, self.load_transformer_path, prefix="transformer.")
        else:
            if self.load_transformer_path:
                self._load_submodule_weights(self.model, self.load_transformer_path, prefix="transformer.")

        # return model
        return self.model, current_epoch, best_epoch, best_loss, input_norm_dict

    # def load_checkpoint(self, checkpoint):

    @utils.initializer
    def train(self,split=(64,16,20)):
        """Train the model and evaluate on train/val/test splits.

        Args:
            split (tuple[int, int, int]): Train/val/test percent split.
        """
        smoke_steps = None
        if self.smoke_train:
            if self.total_epochs < 1:
                self.total_epochs = 1
            else:
                self.total_epochs = min(self.total_epochs, 1)
            self.epoch_early_stop = 1
            self.sample_limit = min(self.sample_limit, 5000)
            smoke_steps = self.smoke_num_steps
            print(
                "smoke_train enabled:",
                f"samples={self.sample_limit}, steps={smoke_steps}, epochs={self.total_epochs}",
            )
        model, current_epoch, best_epoch, best_loss, input_norm_dict = self._setup_model()
        train_dataloader, val_dataloader, test_dataloader, input_norm_dict = self._setup_data(
            input_norm_dict,
            use_train_transforms=True,
        )
        
        self._run_training_loop(
            model,
            current_epoch,
            best_epoch,
            best_loss,
            input_norm_dict,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            smoke_steps=smoke_steps,
        )

    def _run_training_loop(
        self,
        model,
        current_epoch,
        best_epoch,
        best_loss,
        input_norm_dict,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        smoke_steps=None,
        epoch_hook=None,
    ):
        """Run the main training loop with an optional per-epoch hook."""
        print('begin training loop')
        monitor_thread = None
        monitor_stop = None
        if self.gpu_monitor:
            self._gpu_status = ""
            monitor_thread, monitor_stop = self._start_gpu_monitor()
        ram_thread = None
        ram_stop = None
        if self.ram_monitor:
            self._ram_status = ""
            ram_thread, ram_stop = self._start_ram_monitor()
        prof = None
        prof_records = []
        profile_steps = int(self.profile_steps) if self.profile_steps is not None else 0
        profile_steps = max(0, profile_steps)
        if self.profile and profile_steps > 0:
            os.makedirs(self.profile_dir, exist_ok=True)
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profile_dir),
            )
            prof.start()

        def _mem(tag):
            if not torch.cuda.is_available():
                return
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_alloc = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[mem] {tag}: alloc={alloc:.2f}G reserved={reserved:.2f}G max={max_alloc:.2f}G")

        global_step = 0
        prev_batch_end = None
        while (current_epoch < self.total_epochs) and (
            current_epoch - best_epoch < self.epoch_early_stop
        ):
            if epoch_hook is not None:
                epoch_hook(current_epoch)
                self._set_trainable_flags()
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.learning_rate
            if self.panecho_trainable and len(self._panecho_cache) > 0:
                print("clearing cached panecho embeddings (panecho_trainable=True)")
                self._panecho_cache_clear()

            self.model.train()
            epoch_start_time = time.time()
            train_loss_total = 0

            pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {current_epoch}",
                total=len(train_dataloader),
            )
            for batch_count, (Embedding, Correct_Out, EID) in enumerate(pbar):
                postfix = []
                if self.gpu_monitor and self._gpu_status:
                    postfix.append(f"gpu={self._gpu_status}")
                if self.ram_monitor and self._ram_status:
                    postfix.append(f"ram={self._ram_status}")
                if postfix:
                    pbar.set_postfix_str(" ".join(postfix))
                batch_start = time.time()
                data_wait = None
                if prev_batch_end is not None:
                    data_wait = batch_start - prev_batch_end

                if (torch.cuda.is_available()):
                    Embedding = Embedding.to('cuda')
                    Correct_Out = Correct_Out.to('cuda')
                    
                if self.debug_mem and batch_count == 0:
                    _mem("before forward")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_fwd_start = time.time()
                with torch.amp.autocast("cuda", enabled=self.amp):
                    if self.end_to_end and self.cache_panecho_embeddings and not self.panecho_trainable:
                        eid_val = EID[0] if isinstance(EID, (list, tuple, np.ndarray)) else EID
                        try:
                            eid_key = int(eid_val)
                        except Exception:
                            eid_key = str(eid_val)
                        cached = self._panecho_cache_get(eid_key)
                        if cached is not None:
                            emb = cached.to(Embedding.device)
                        else:
                            with torch.no_grad():
                                emb = self.model._panecho_embed(Embedding)
                            self._panecho_cache_put(eid_key, emb.detach().cpu())
                        out = self.model.transformer(emb)
                    else:
                        out = self.model(Embedding)
                    train_loss = self.loss_fn(out, Correct_Out)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_fwd_end = time.time()
                if self.debug_mem and batch_count == 0:
                    _mem("after forward")
                if self.debug_mem and batch_count == 0:
                    _mem("after loss")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_bwd_start = time.time()
                if self.amp:
                    self.scaler.scale(train_loss).backward()
                else:
                    train_loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_bwd_end = time.time()
                if self.debug_mem and batch_count == 0:
                    _mem("after backward")
                train_loss_total += train_loss.item()
                
                step_performed = False
                if ( (batch_count+1) % self.batch_number ==0) :           
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_step_start = time.time()
                    if self.amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.model.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_step_end = time.time()
                    step_performed = True
                elif ( (batch_count+1) == len(train_dataloader) ):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_step_start = time.time()
                    if self.amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.model.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_step_end = time.time()
                    step_performed = True

                if smoke_steps is not None and (batch_count + 1) >= smoke_steps:
                    if (batch_count + 1) % self.batch_number != 0:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t_step_start = time.time()
                        if self.amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.model.zero_grad()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t_step_end = time.time()
                        step_performed = True
                    break
                
                if prof is not None and global_step < profile_steps:
                    prof.step()
                if prof is not None and global_step == profile_steps:
                    if self.profile_summary:
                        prof_records.append(prof)
                    prof.stop()
                    prof = None
                
                if self.timing_every and (batch_count + 1) % int(self.timing_every) == 0:
                    step_time = (t_step_end - t_step_start) if step_performed else 0.0
                    total_time = time.time() - batch_start
                    data_wait_s = data_wait if data_wait is not None else 0.0
                    print(
                        f"[timing] batch={batch_count+1} data_wait={data_wait_s:.2f}s "
                        f"fwd={t_fwd_end - t_fwd_start:.2f}s bwd={t_bwd_end - t_bwd_start:.2f}s "
                        f"step={step_time:.2f}s total={total_time:.2f}s"
                    )
                prev_batch_end = time.time()
                global_step += 1
                    
            epoch_end_time = time.time()
            
            __, __, __, val_loss_total = run_model_on_dataloader(
                self.model,
                val_dataloader,
                self.loss_fn,
                amp=self.amp,
            )
            
            current_epoch = current_epoch + 1
            
            tmp_LR = self.optimizer.state_dict()['param_groups'][0]['lr']
            perf = {
                    'epoch':current_epoch,
                    'train loss':train_loss_total,
                    'val loss':val_loss_total,
                    'lr':tmp_LR,
                    'epoch time':epoch_end_time - epoch_start_time,
            }
            self.perf_log.append(list(perf.values()))
            print(' '.join([f'{k}: {v}' for k,v in perf.items() if k in ['train loss','val loss','lr']]))
            self.save_log()
            
            self.scheduler.step(val_loss_total)
            save_nn(
                self.model,
                self.last_checkpoint_path,
                self.perf_log,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                input_norm_dict=input_norm_dict,
            )
                
            if (val_loss_total < best_loss):
                save_nn(
                    self.model,
                    self.best_checkpoint_path,
                    self.perf_log,
                    optimizer=None,
                    scheduler=None,
                    input_norm_dict=input_norm_dict,
                )
                best_loss = val_loss_total 
                best_epoch = current_epoch

            if (current_epoch == self.total_epochs): 
                print('current epoch = epoch limit, terminating')
            if current_epoch - best_epoch == self.epoch_early_stop:
                print('early stopping')

            
        if prof is not None:
            if self.profile_summary:
                prof_records.append(prof)
            prof.stop()
            prof = None

        if self.profile_summary and prof_records:
            try:
                summary = prof_records[-1].key_averages().table(
                    sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
                    row_limit=50,
                )
                print("Profiler summary (top ops):")
                print(summary)
            except Exception as e:
                print(f"warning: failed to print profiler summary: {e}")

        if monitor_stop is not None:
            monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
        if ram_stop is not None:
            ram_stop.set()
        if ram_thread is not None:
            ram_thread.join(timeout=1.0)

        utils.plot_training_progress( self.model_path, self.perf_log)
        print('Training Completed')
        
        best_checkpoint_path = os.path.join(self.model_path, 'best_checkpoint.pt') 
        best_model,_,_,_,input_norm_dict  = load_model_and_random_state(best_checkpoint_path, model)
        for dataloader,fold in zip((train_dataloader,val_dataloader,test_dataloader),('train','val','test')):
            self._evaluate(best_model, dataloader, fold, input_norm_dict)

    @utils.initializer
    def train_ping_pong(
        self,
        total_epochs=10,
        start_with="transformer",
        switch_every=1,
        transformer_lr=None,
        panecho_lr=None,
        split=(64,16,20),
    ):
        """Alternate training between transformer-only and PanEcho-only phases.

        Args:
            total_epochs (int): Total epochs to train to.
            start_with (str): "transformer" or "panecho".
            switch_every (int): Number of epochs per phase.
            transformer_lr (float|None): Optional LR for transformer-only phases.
            panecho_lr (float|None): Optional LR for panecho-only phases.
            split (tuple[int,int,int]): Train/val/test split.
        """
        if start_with not in ("transformer", "panecho"):
            raise ValueError("start_with must be 'transformer' or 'panecho'")
        smoke_steps = None
        if self.smoke_train:
            if total_epochs < 1:
                total_epochs = 1
            self.epoch_early_stop = 1
            self.sample_limit = min(self.sample_limit, 5000)
            smoke_steps = self.smoke_num_steps
            print(
                "smoke_train enabled:",
                f"samples={self.sample_limit}, steps={smoke_steps}, epochs={total_epochs}",
            )

        self.total_epochs = int(total_epochs)
        model, current_epoch, best_epoch, best_loss, input_norm_dict = self._setup_model()
        train_dataloader, val_dataloader, test_dataloader, input_norm_dict = self._setup_data(
            input_norm_dict,
            use_train_transforms=True,
        )

        orig_lr = self.learning_rate
        switch_every = max(1, int(switch_every))
        start_phase_idx = 0 if start_with == "transformer" else 1

        def _epoch_hook(epoch):
            phase_idx = ((epoch // switch_every) + start_phase_idx) % 2
            if phase_idx == 0:
                print(f"phase: transformer-only epoch {epoch}")
                self.panecho_trainable = False
                self.transformer_trainable = True
                if transformer_lr is not None:
                    self.learning_rate = transformer_lr
                else:
                    self.learning_rate = orig_lr
            else:
                print(f"phase: panecho-only epoch {epoch}")
                self.panecho_trainable = True
                self.transformer_trainable = False
                if panecho_lr is not None:
                    self.learning_rate = panecho_lr
                else:
                    self.learning_rate = orig_lr

        self._run_training_loop(
            model,
            current_epoch,
            best_epoch,
            best_loss,
            input_norm_dict,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            smoke_steps=smoke_steps,
            epoch_hook=_epoch_hook,
        )
    def _evaluate(self, model, dataloader, fold, input_norm_dict=None):
        """Evaluate a model on a dataloader and write outputs.

        Args:
            model (torch.nn.Module): Trained model.
            dataloader (torch.utils.data.DataLoader): Dataloader for a split.
            fold (str): Split name (train/val/test).
            input_norm_dict (dict|None): Normalization parameters.
        """
        if dataloader is None or len(dataloader)==0:
            print('skipping',fold)
            return
        # Run on test dataset
        print(f'run model on {fold} set')
        def _cache_hook(embedding, eid):
            if self.end_to_end and self.cache_panecho_embeddings and not self.panecho_trainable:
                eid_val = eid[0] if isinstance(eid, (list, tuple, np.ndarray)) else eid
                try:
                    eid_key = int(eid_val)
                except Exception:
                    eid_key = str(eid_val)
                cached = self._panecho_cache_get(eid_key)
                if cached is not None:
                    emb = cached.to(embedding.device)
                else:
                    emb = model._panecho_embed(embedding)
                    self._panecho_cache_put(eid_key, emb.detach().cpu())
                return model.transformer(emb)
            return model(embedding)

        y_true, y_pred, EIDs, loss = run_model_on_dataloader(
            model,
            dataloader,
            self.loss_fn,
            amp=self.amp,
            cache_hook=_cache_hook,
        )
        # convert model outputs back
        y_true = np.array(y_true).squeeze()
        # y_true_test_norm = return_correct_output_np
        y_pred = np.array(y_pred)
        if self.task == 'measure':
            y_true = utils.un_normalize_output(y_true, self.task_labels, input_norm_dict)
            y_pred = utils.un_normalize_output(y_pred, self.task_labels, input_norm_dict)

        saveout_path = os.path.join(self.model_path, f'saveout_{fold}_{self.dataset}.csv')
        if isinstance(dataloader.dataset,torch.utils.data.Subset):
            data_df = dataloader.dataset.dataset.dataframe
        else:
            data_df = dataloader.dataset.dataframe
        PIDs = data_df.loc[EIDs]['pid'].values
        EIDs = np.array(EIDs)
        saveout_df = pd.DataFrame({'PID':PIDs,'Echo_ID':EIDs})
        # saveout_df['PID'] = PIDs
        # saveout_df['Echo_ID'] = EIDs
        for i,k in enumerate(self.task_labels):
            saveout_df[self.task_labels[i]+'_Correct'] = y_true[:,i]
            saveout_df[self.task_labels[i]+'_Predict'] = y_pred[:,i]
        print('writing',saveout_path)
        saveout_df.to_csv(saveout_path)

        if self.task == 'measure': 
            utils.scatter_plots(self.model_path, self.dataset, fold, self.task_labels, y_true, y_pred)

    def _bootstrap_metric_ci(
        self,
        y_true,
        y_pred,
        metric_fn,
        rng,
        n_bootstrap=1000,
        ci_percentiles=(2.5, 97.5),
        require_two_classes=False,
    ):
        """Compute metric value and bootstrap confidence interval."""
        if y_true.size == 0:
            return None, None, None, 0
        if require_two_classes and np.unique(y_true).size < 2:
            return None, None, None, 0

        try:
            metric_value = float(metric_fn(y_true, y_pred))
        except ValueError:
            metric_value = None

        bootstrap_values = []
        n = y_true.shape[0]
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, n, size=n)
            yb_true = y_true[idx]
            yb_pred = y_pred[idx]
            if require_two_classes and np.unique(yb_true).size < 2:
                continue
            try:
                val = metric_fn(yb_true, yb_pred)
            except ValueError:
                continue
            if np.isfinite(val):
                bootstrap_values.append(float(val))

        if len(bootstrap_values) == 0:
            return metric_value, None, None, 0

        low, high = np.percentile(np.array(bootstrap_values), list(ci_percentiles))
        return metric_value, float(low), float(high), len(bootstrap_values)

    def get_metrics(self, fold='test', dataset=None, n_bootstrap=1000, model_name=None):
        """Compute task metrics from saved predictions and write JSON output."""
        if model_name is not None:
            self.model_name = model_name
            self.model_path = os.path.join('./trained_models', model_name)

        if dataset is None:
            dataset = self.dataset
        saveout_path = os.path.join(self.model_path, f'saveout_{fold}_{dataset}.csv')
        if not os.path.exists(saveout_path):
            print(f'WARNING: saveout file not found, skipping metrics: {saveout_path}')
            return None

        df = pd.read_csv(saveout_path)
        rng = np.random.default_rng(self.seed)
        out = {
            'task': self.task,
            'dataset': dataset,
            'fold': fold,
            'n_bootstrap': int(n_bootstrap),
            'metrics': {},
        }

        y_true_all = []
        y_pred_all = []
        valid_labels = []

        for label in self.task_labels:
            true_col = f'{label}_Correct'
            pred_col = f'{label}_Predict'
            if true_col not in df.columns or pred_col not in df.columns:
                print(f'WARNING: missing columns for label {label}, skipping')
                continue

            y_true = pd.to_numeric(df[true_col], errors='coerce').to_numpy(dtype=float)
            y_pred = pd.to_numeric(df[pred_col], errors='coerce').to_numpy(dtype=float)
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

            label_metrics = {'n_samples': int(y_true.shape[0])}
            if self.task in ['chd', 'fyler']:
                roc_val, roc_lo, roc_hi, roc_eff_n = self._bootstrap_metric_ci(
                    y_true,
                    y_pred,
                    roc_auc_score,
                    rng=rng,
                    n_bootstrap=n_bootstrap,
                    require_two_classes=True,
                )
                ap_val, ap_lo, ap_hi, ap_eff_n = self._bootstrap_metric_ci(
                    y_true,
                    y_pred,
                    average_precision_score,
                    rng=rng,
                    n_bootstrap=n_bootstrap,
                    require_two_classes=True,
                )
                label_metrics['roc_auc_score'] = {
                    'value': roc_val,
                    'ci_lower': roc_lo,
                    'ci_upper': roc_hi,
                    'n_bootstrap_effective': roc_eff_n,
                }
                label_metrics['average_precision'] = {
                    'value': ap_val,
                    'ci_lower': ap_lo,
                    'ci_upper': ap_hi,
                    'n_bootstrap_effective': ap_eff_n,
                }
                y_true_all.append(y_true)
                y_pred_all.append(y_pred)
                valid_labels.append(label)
            elif self.task == 'measure':
                mae_val, mae_lo, mae_hi, mae_eff_n = self._bootstrap_metric_ci(
                    y_true,
                    y_pred,
                    median_absolute_error,
                    rng=rng,
                    n_bootstrap=n_bootstrap,
                )
                r2_val, r2_lo, r2_hi, r2_eff_n = self._bootstrap_metric_ci(
                    y_true,
                    y_pred,
                    r2_score,
                    rng=rng,
                    n_bootstrap=n_bootstrap,
                )
                label_metrics['median_absolute_error'] = {
                    'value': mae_val,
                    'ci_lower': mae_lo,
                    'ci_upper': mae_hi,
                    'n_bootstrap_effective': mae_eff_n,
                }
                label_metrics['r2_score'] = {
                    'value': r2_val,
                    'ci_lower': r2_lo,
                    'ci_upper': r2_hi,
                    'n_bootstrap_effective': r2_eff_n,
                }
            out['metrics'][label] = label_metrics

        metrics_path = os.path.join(self.model_path, f'metrics_{fold}_{dataset}.json')
        with open(metrics_path, 'w') as f:
            json.dump(out, f, indent=2)
        print('writing', metrics_path)

        if self.task in ['chd', 'fyler'] and len(valid_labels) > 0:
            max_n = max(arr.shape[0] for arr in y_true_all)
            y_true_mat = np.full((max_n, len(valid_labels)), np.nan)
            y_pred_mat = np.full((max_n, len(valid_labels)), np.nan)
            for i, (yt, yp) in enumerate(zip(y_true_all, y_pred_all)):
                y_true_mat[: yt.shape[0], i] = yt
                y_pred_mat[: yp.shape[0], i] = yp
            utils.plot_roc_curves(self.model_path, dataset, fold, valid_labels, y_true_mat, y_pred_mat)
            utils.plot_pr_curves(self.model_path, dataset, fold, valid_labels, y_true_mat, y_pred_mat)
        return out
        
            
    def evaluate(self):
        """Evaluate the best checkpoint on train/val/test splits."""
        # 10. Compute performance 
        eval_start_time = time.time()
        # model = self._setup_model() 
        # best_checkpoint_path = os.path.join(self.model_path, 'best_checkpoint.pt') 
        # best_model, _, _, _, self.input_norm_dict = load_model_and_random_state(best_checkpoint_path, model)

        model,_,_,_,_ = self._setup_model() 
        best_checkpoint_path = os.path.join(self.model_path, 'best_checkpoint.pt') 
        best_model,_,_,_,input_norm_dict  = load_model_and_random_state(best_checkpoint_path, model)
        train_dl,val_dl,test_dl,input_norm_dict = self._setup_data(
            input_norm_dict,
            use_train_transforms=False,
        )

        for dataloader, fold in zip((train_dl, val_dl, test_dl), ('train', 'val', 'test')):
            self._evaluate(best_model, dataloader, fold, input_norm_dict)
            self.get_metrics(fold=fold, dataset=self.dataset)
            print('eval time taken: ', time.time() - eval_start_time)

    def _set_loss(self):
        """Set the loss function based on task type."""
        if self.task=='measure':
            self.loss_fn = utils.masked_mse_loss
        elif self.task in ['chd','fyler']:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

    @utils.initializer
    def embed(self, embed_file=None, split=(0,0,100)):
        """Generate and save embedding vectors for each study.

        Args:
            embed_file (str|None): Output HDF5 path; defaults to model directory.
            split (tuple[int, int, int]): Train/val/test split (unused).
        """
        model,_,_,_,_ = self._setup_model() 
        best_checkpoint_path = os.path.join(self.model_path, 'best_checkpoint.pt') 
        best_model,_,_,_,input_norm_dict  = load_model_and_random_state(best_checkpoint_path, model)
        _,_,dataloader,input_norm_dict = self._setup_data(
            input_norm_dict,
            use_train_transforms=False,
        )
        
        if embed_file is None:
            embed_file = os.path.join(self.model_path,f'embeddings_{self.dataset}_{self.task}.h5')
        with h5py.File(embed_file, 'w') as f:
            f.attrs['dataset'] = self.dataset
            f.attrs['model_name'] = self.model_name
            f.attrs['task'] = self.task
            pbar = tqdm(dataloader, total=len(dataloader.dataset))
            for embedding, correct_labels, eid in pbar:
                if torch.cuda.is_available():
                    embedding = embedding.to("cuda")

                with torch.no_grad():
                    f[str(eid)] = model.embed(embedding).cpu().numpy()
        print(f'saved embeddings to {embed_file}')

    @utils.initializer
    def explain(
        self,
        explain=False,
        explain_n=5,
        explain_mode='pred',
        explain_tasks = ('EF05','AR01'),
        split=(0,0,100)
    ):
        """Generate integrated gradients explanations and save to CSV.

        Args:
            explain (bool): Unused flag for CLI compatibility.
            explain_n (int): Number of top videos to record per sample.
            explain_mode (str): Objective mode for IG ("pred" or "loss").
            explain_tasks (tuple[str]|str): Tasks to explain.
            split (tuple[int, int, int]): Train/val/test split (unused).
        """
        from integrated_gradients import integrated_gradients_video_level
        if isinstance(self.explain_tasks,str):
            if self.explain_tasks.lower() == 'all':
                self.explain_tasks = self.task_labels
            else:
                self.explain_tasks = tuple(self.explain_tasks)
        print('explain_tasks:',self.explain_tasks)
        model,_,_,_,_ = self._setup_model() 
        best_checkpoint_path = os.path.join(self.model_path, 'best_checkpoint.pt') 
        best_model,_,_,_,input_norm_dict  = load_model_and_random_state(best_checkpoint_path, model)
        _,_,test_dataloader,input_norm_dict = self._setup_data(
            input_norm_dict,
            use_train_transforms=False,
        )
        
        # Run on test dataset
        print('run model on test set')
        return_model_outputs, return_correct_outputs, return_EIDs, loss = run_model_on_dataloader(
            best_model,
            test_dataloader,
            self.loss_fn,
            amp=self.amp,
        )
        # convert model outputs back
        return_correct_output_np = np.array(return_correct_outputs).squeeze()
        y_true_test_norm = return_correct_output_np
        return_model_output_np = np.array(return_model_outputs)
        if self.task == 'measure':
            return_model_output_np = utils.un_normalize_output(return_model_output_np, self.task_labels, input_norm_dict)
            return_correct_output_np = utils.un_normalize_output(return_correct_output_np, self.task_labels, input_norm_dict)
        y_pred_test = return_model_output_np
        y_true_test = return_correct_output_np

        # pick samples as follows:
        # quintiles of measurement
        # within each quintile, samples with error < mae 
        # pick 10 random samples there
        # choose a random sample
        frames = []
        # measures = ['EF05', 'AR01']
        measure_maes = {'EF05':0.0277,'AR01':0.13}
        for measure in self.explain_tasks:
            task_idx=self.task_labels.index(measure)
            y_trues = y_true_test[:,task_idx]
            y_trues_norm = y_true_test_norm[:,task_idx]
            y_preds = y_pred_test[:,task_idx]
            if self.task != 'measure':
                # apply sigmoid to logits for classifier outputs
                y_preds = [utils.sigmoid(yp) for yp in y_preds]
            if self.task == 'measure':
                quantiles = [0.]+[
                    np.nanquantile(y_trues,i) for i in [.2, .4, .6, .8, 1.]
                    # np.nanquantile(y_trues,i) for i in [0.5, 1.]
                ]
                test_errors = np.abs(y_trues-y_preds)
                sample_size=10
            else:
                # quantiles = [-1.,0.5,1.]
                # only positive samples
                quantiles = [0.5,1.] 
                # import sklearn.metrics 
                # test_errors = sklearn.metrics.log_loss(y_trues,y_preds)
                y_pred_top100 = np.sort(y_preds)[-100:][0]
                sample_size=50
            print('quantiles for',measure,':',quantiles)
            sample_idxs = np.arange(len(y_trues))
            for q_bot,q_top in zip(quantiles[:-1],quantiles[1:]):
                if self.task == 'measure':
                    mask = (
                        (~np.isnan(y_trues))
                        & (y_trues > q_bot) 
                        & (y_trues <= q_top) 
                        & (test_errors < measure_maes[measure])
                    )
                else:
                    mask = (
                        (~np.isnan(y_trues))
                        & (y_trues > q_bot) # true labels 
                        & (y_trues <= q_top) 
                        & (y_preds > y_pred_top100) # nominally positive classifications
                    )
                sample_idxs_subset = sample_idxs[mask]
                # sample_size = int(50/(len(quantiles)-1))
                if len(sample_idxs_subset) <= sample_size:
                    print('not enough samples (len subset:',len(sample_idxs_subset),')', 'sample_size:',sample_size)
                    print('there are ',(~np.isnan(y_trues)).sum(),'non-missing labels')
                    print('there are ',((y_trues > q_bot) & (y_trues <= q_top)).sum(),f'samples in [{q_bot},{q_top}]')
                    print('there are',(y_preds > y_pred_top100).sum(),'predictions >',y_pred_top100)
                    print('try relaxing y_pred_top100 constraint')
                    mask = (
                        (~np.isnan(y_trues))
                        & (y_trues > q_bot) # true labels 
                        & (y_trues <= q_top) 
                        # & (y_preds > y_pred_top100) # nominally positive classifications
                    )
                    sample_idxs_subset = sample_idxs[mask]
                    if len(sample_idxs_subset) <= sample_size:
                        print('didnt work, adjusting samples to',len(sample_idxs_subset))
                        sample_size = len(sample_idxs_subset)
                        print('new sample size:',sample_size)
                assert len(sample_idxs_subset) >= sample_size, "not enough samples per quantile" 
                chosen_idxs = np.random.choice(sample_idxs_subset, size=sample_size, replace=False)
                for i in chosen_idxs:
                    sample = test_dataloader.dataset[i]
                    y_true = y_trues[i]
                    y_true_norm = y_trues_norm[i]
                    y_pred = y_preds[i]
                    if isinstance(test_dataloader.dataset,torch.utils.data.Subset):
                        study_filenames, echo_id = test_dataloader.dataset.dataset.get_filenames(i)
                    else:
                        study_filenames, echo_id = test_dataloader.dataset.get_filenames(i)
                    x_list, y, idx = sample #["videos"], sample["target"]
                    y_norm = y.cpu().numpy().T
                    if self.task=='measure':
                        y = utils.un_normalize_output(y_norm, self.task_labels, input_norm_dict)
                    y = y.reshape(-1)
                    assert y[task_idx] == y_true # sanity check
                    assert not np.isnan(y_true)

                    scores, attrs, obj, yhat = integrated_gradients_video_level(
                        best_model,
                        x_list,
                        mode=explain_mode,
                        # mode="loss",
                        loss='mae' if self.task == 'measure' else 'bce_logits',
                        y_true=y_true_norm,
                        task_type="regression" if self.task == 'measure' else 'classification',
                        task_idx=task_idx,
                        steps=64,
                    )
                    scores = scores.cpu().numpy()
                    cap = min(x_list.shape[0], explain_n)
                    ind = np.argpartition(scores, -cap)[-cap::-1]
                    top5scores = scores[ind]
                    top_filenames = study_filenames[ind]
                    #use logistic loss for class, abs error for regression
                    if self.task!='measure': 
                        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1-y_pred))
                    else:
                        loss=np.abs(y_true - y_pred)
                    result = dict(
                        measure=measure,
                        echo_id=echo_id,
                        y_pred=y_pred ,
                        y_true=y_true,
                        loss=loss
                    )

                    for k in np.arange(explain_n):
                        if k >= len(top_filenames):
                            result[f'top_video_{k+1}'] = None
                            result[f'top_video_{k+1}_score'] = None
                        else:
                            result[f'top_video_{k+1}'] = top_filenames[k]
                            result[f'top_video_{k+1}_score'] = top5scores[k]
                    frames.append(result)
        df_explain = pd.DataFrame(frames)
        tv_cols = [c for c in df_explain.columns if 'top_video' in c and 'score' not in c]
        tvs_cols = [c for c in df_explain.columns if 'top_video' in c and 'score' in c]
        df_explain = df_explain[['echo_id','measure','y_true','y_pred','loss']+tv_cols+tvs_cols]
        df_explain['dataset'] = self.dataset
        explain_file_name = f'explanation_test_{self.dataset}.explain_n-{explain_n}.mode-{explain_mode}.csv'
        df_explain.to_csv(os.path.join(self.model_path, explain_file_name))
        print('saved explanations to',os.path.join(self.model_path, explain_file_name))

    def save_log(self):
        """Write the training loss log to CSV."""
        # save model runtime and loss as csv
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epochs_trained", "train_loss", "val_loss", "lr", "epoch_time"]
            )
            writer.writerows(self.perf_log)


def save_nn(model, path, perf_log, optimizer=None, scheduler=None, input_norm_dict=None):
    """Save a model checkpoint to disk.

    Args:
        model (torch.nn.Module): Model to save.
        path (str): Checkpoint path.
        perf_log (list[list]): Training log entries.
        optimizer (torch.optim.Optimizer|None): Optimizer state to save.
        scheduler (torch.optim.lr_scheduler._LRScheduler|None): Scheduler state to save.
        input_norm_dict (dict|None): Normalization parameters.
    """
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    # best_performance_measure refers to the performance of the best model so far
    # so we don't accidentally overwrite it

    out_dict = {}

    out_dict["model_state_dict"] = model.state_dict()
    out_dict["perf_log"] = perf_log

    out_dict["numpy_random_state"] = np.random.get_state()
    out_dict["torch_random_state"] = torch.get_rng_state()
    out_dict["cuda_random_state"] = torch.cuda.get_rng_state()

    if optimizer is not None:
        out_dict["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        out_dict["scheduler_state_dict"] = scheduler.state_dict()

    if input_norm_dict is not None: # stores normalization type, measures, and param/mean/stdev per measure 
        out_dict['input_norm_dict'] = input_norm_dict
    torch.save(out_dict, path)



def load_model_and_random_state(path, model, optimizer=None, scheduler=None):
    """Load a checkpoint and restore model and RNG state.

    Args:
        path (str): Checkpoint path.
        model (torch.nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer|None): Optimizer to restore.
        scheduler (torch.optim.lr_scheduler._LRScheduler|None): Scheduler to restore.

    Returns:
        tuple: (model, optimizer, scheduler, perf_log, input_norm_dict)
    """
    # input: .pt location
    # do: pull model, pull training progress, set random states
    # output: model, training progress

    import_dict = torch.load(path, weights_only=False)  # load a checkpoint

    model.load_state_dict(import_dict["model_state_dict"])
    if "optimizer_state_dict" in import_dict.keys() and (optimizer is not None):
        optimizer.load_state_dict(import_dict["optimizer_state_dict"])
    else:
        print("warning no optimizer loaded")
    if "scheduler_state_dict" in import_dict.keys() and (scheduler is not None):
        scheduler.load_state_dict(import_dict["scheduler_state_dict"])
    else:
        print("warning no scheduler loaded")

    utils.load_random_state(import_dict)
    perf_log = import_dict["perf_log"]

    print("model loaded, epoch", perf_log[-1][0])

    if ('input_norm_dict' in import_dict.keys()):
        input_norm_dict = import_dict['input_norm_dict'] # pull normalization details and parameters
        print('Loaded input_norm_dict')
    else:
        input_norm_dict = None
        print('input_norm_dict NOT loaded')
    
    return model, optimizer, scheduler, perf_log, input_norm_dict


def run_model_on_dataloader(model, dataloader, loss_func_pointer, amp=False, cache_hook=None):
    """Run inference on a dataloader and collect outputs.

    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader to iterate.
        loss_func_pointer (callable): Loss function to compute per batch.

    Returns:
        tuple: (model_outputs, correct_outputs, echo_ids, total_loss)
    """
    # runs model on dataloader, measuring loss and returning  correct and output values and pid (folder) and loss

    model.eval()
    return_model_outputs = []
    return_correct_outputs = []
    return_EIDs = []
    loss = 0
    pbar = tqdm(dataloader, total=len(dataloader.dataset), desc="Inference")
    use_amp = amp and torch.cuda.is_available()
    for embedding, correct_labels, eid in pbar:
        if torch.cuda.is_available():
            embedding = embedding.to("cuda")
            correct_labels = correct_labels.to("cuda")

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                if cache_hook is not None:
                    model_outputs = cache_hook(embedding, eid)
                else:
                    model_outputs = model(embedding)
            return_model_outputs.append(model_outputs.to('cpu'))
            return_correct_outputs.append(correct_labels.to('cpu'))
            return_EIDs.append(eid)
            loss += float(loss_func_pointer(model_outputs, correct_labels).to("cpu"))
# 
    return return_model_outputs, return_correct_outputs, return_EIDs, loss


    

if __name__ == "__main__":
    fire.Fire(EchoFocus)
