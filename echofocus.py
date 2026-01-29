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

# import cv2
import numpy as np
# from torchvision import tv_tensors
# from torchvision.transforms import resize, center_crop

# from torchvision.transforms import v2


from tqdm import tqdm
import uuid

import utils
from datasets import CustomDataset, get_dataset, custom_collate
from models import CustomTransformer


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
        epoch_lim=-1,
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
        cache_embeddings=False
    ):
        """Initialize training/evaluation state and load config.

        Args:
            model_name (str|None): Name for the model run directory.
            dataset (str|None): Dataset key in the config file.
            task (str): Task key in the config file.
            seed (int): RNG seed for reproducibility.
            batch_number (int): Gradient accumulation steps.
            batch_size (int): Batch size (only 1 supported).
            epoch_lim (int): Max epochs to train; -1 for eval-only.
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
            preload_embeddings (bool): Deprecated preload mode.
            run_id (str|None): Optional run ID for reproducibility.
            config (str): Path to config JSON file.
            cache_embeddings (bool): Cache embeddings in memory.
        """
        self.time = time.time()
        self.datetime = str(datetime.now()).replace(" ", "_")
        if run_id:
            self.run_id = run_id
        else:
            self.run_id = f"{self.datetime}_{uuid.uuid4()}"

        assert batch_size==1, "only batch_size=1 currently supported"
        print('main')
        args = {**locals()}
        # input is paired dict of strings named args
        start_time = time.time()

        print("random seed", seed, "\n")
        print("batch_number", batch_number, "\n")

        if epoch_lim == -1:
            print("epoch lim missing. evaluating model")
        print("epoch_lim", epoch_lim, "\n")

        if epoch_early_stop == 9999:
            print("no early stop. defaulting to 10k epochs")
        print("epoch_early_stop", epoch_early_stop, "\n")

        print("learning_rate", learning_rate, "\n")


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
        torch.backends.cudnn.deterministic = True # make TRUE if you want reproducible results (slower)     
         
        # set model name 
        if not model_name:
            model_name = f'{task}_{self.run_id}'
        self.model_path = os.path.join('./trained_models', model_name)
        os.makedirs(self.model_path,exist_ok=True)

        self._load_config()
        self._set_loss()

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
            setattr(self,k,v)
        for k,v in data['dataset'][self.dataset].items():
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

    def _setup_data(self, input_norm_dict=None):
        """Prepare dataloaders and normalization metadata.

        Args:
            input_norm_dict (dict|None): Existing normalization parameters.

        Returns:
            tuple: (train_dataloader, valid_dataloader, test_dataloader, input_norm_dict)
        """
        csv_data = pd.read_csv(self.label_path) # pull labels from local path
        csv_data = csv_data.drop_duplicates() # I don't know why there are duplicates, but there are...
        Embedding_EchoID_List = [int(k.split('_')[0]) for k in os.listdir(self.embedding_path)]

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
        study_embeddings = get_dataset(
            self.embedding_path,
            eid_keep_list,
            limit=self.sample_limit,
            parallel_processes=self.parallel_processes,
            # preload=self.preload_embeddings,
            cache_embeddings=self.cache_embeddings,
            batch_size=self.batch_size
        )
        # print('Total videos included: ',sum([study_embeddings[key].shape[0] for key in study_embeddings.keys()]))
        # so study_embeddings is a dict of M x 16 x 768, indexed by echo ID (EID)
        
        #because of laptop_debug we don't always keep all the eids. limit the dataframe to what we pulled
        if self.preload_embeddings:
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
        # import ipdb                
        # ipdb.set_trace()
        # self._setup_model()

        # 7. Get normalization parameters, normalize datasets
        # if (('input_norm_dict' not in locals()) or (input_norm_dict is None)): # if didn't get or never had
        if self.task=='measure':
            if input_norm_dict is None:
                print('no input_norm_dict loaded, generating from Train_DF')
                input_norm_dict = utils.get_norm_params(Train_DF, self.task_labels)
            Train_DF = utils.normalize_df(Train_DF,input_norm_dict)
            Valid_DF = utils.normalize_df(Valid_DF,input_norm_dict)
            Test_DF = utils.normalize_df(Test_DF,input_norm_dict)
            print('normalized labels')

        test_dataset = CustomDataset(Test_DF, study_embeddings, self.task_labels) #, study_filenames)
        if self.sample_limit < len(test_dataset):
            print('subsampling test dataset')
            test_dataset = torch.utils.data.Subset(test_dataset, list(range(0, self.sample_limit)))
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=custom_collate,
            # num_workers=self.parallel_processes
        )

        if (Tr == 0):
            return None, None, test_dataloader, input_norm_dict
        else:
            # weights = np.ones(len(Train_DF))
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
                # num_workers=self.parallel_processes
            )
            
            valid_dataset = CustomDataset(Valid_DF, study_embeddings, self.task_labels) #, study_filenames)
            if self.sample_limit < len(valid_dataset):
                print('subsampling valid dataset')
                valid_dataset = torch.utils.data.Subset(valid_dataset, list(range(0, self.sample_limit)))
            valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=custom_collate,
                # num_workers=self.parallel_processes
            )
        

        return train_dataloader, valid_dataloader, test_dataloader, input_norm_dict

    # def _normalize_data(self):

    def _setup_model(self):
        """Initialize model, optimizer, scheduler, and load checkpoints.

        Returns:
            tuple: (model, current_epoch, best_epoch, best_loss, input_norm_dict)
        """
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
        self.model = CustomTransformer(
            input_size=768,
            encoder_dim=768,
            n_encoder_layers=self.encoder_depth,
            output_size=len(self.task_labels),
            clip_dropout=self.clip_dropout,
            tf_combine="avg",
        )
    
        if (torch.cuda.is_available()):
            self.model = self.model.to('cuda')

        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.learning_rate, weight_decay = 0.01)
        # add the scheduler
        patience = 3
        lr_factor = 0.5
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, factor=lr_factor)

        self.perf_log = []
        input_norm_dict=None
        if (os.path.isfile(self.last_checkpoint_path)):
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
            print('no existing lastcheckpoint')

        # return model
        return self.model, current_epoch, best_epoch, best_loss, input_norm_dict

    # def load_checkpoint(self, checkpoint):

    @utils.initializer
    def train(self,split=(64,16,20)):
        """Train the model and evaluate on train/val/test splits.

        Args:
            split (tuple[int, int, int]): Train/val/test percent split.
        """
        model, current_epoch, best_epoch, best_loss, input_norm_dict = self._setup_model()
        train_dataloader, val_dataloader, test_dataloader, input_norm_dict = self._setup_data(input_norm_dict)        
        
        # 9. Train
        # Training loop
        while (current_epoch < self.epoch_lim) and (
            current_epoch - best_epoch < self.epoch_early_stop
        ):
            self.model.train()
            epoch_start_time = time.time()
            # Train an epoch
            train_loss_total = 0

            for batch_count, (Embedding, Correct_Out, EID) in tqdm(
                enumerate(train_dataloader),
                desc=f"Epoch {current_epoch}",
                total=len(train_dataloader),
            ):
                #TODO: for quasi-batch for parallel workers in train data loader,
                # add an extra loop to loop over batches                 

                if (torch.cuda.is_available()):
                    Embedding = Embedding.to('cuda')
                    Correct_Out = Correct_Out.to('cuda')
                    
                out = self.model(Embedding)
                
                train_loss = self.loss_fn(out, Correct_Out)
                # train_loss = Loss_Func(out, Correct_lvef)
                train_loss.backward()
                train_loss_total += train_loss.item()
                
                # Gradient accumulation
                if ( (batch_count+1) % self.batch_number ==0) : # update after batch_number patients            
                    self.optimizer.step()
                    self.model.zero_grad()
                    # print('DEBUG: model updated', (batch_count+1), batch_number)
                    
                elif ( (batch_count+1) == len(train_dataloader) ): # or if on last set of videos
                    self.optimizer.step()
                    self.model.zero_grad()
                    # print('DEBUG: model updated at end of epoch without reaching batch_num', (batch_count+1), len(train_dataloader))
                    
            epoch_end_time = time.time()
            
            # Get loss on validation dataset
            __, __, __, val_loss_total = run_model_on_dataloader(self.model, val_dataloader, self.loss_fn)
            
            # update trained epoch count and log performance
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
            # print(self.perf_log[-1])
            print(' '.join([f'{k}: {v}' for k,v in perf.items() if k in ['train loss','val loss','lr']]))
            self.save_log()
            

            # update scheduler
            self.scheduler.step(val_loss_total)
            save_nn(
                self.model,
                self.last_checkpoint_path,
                self.perf_log,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                input_norm_dict=input_norm_dict,
            )
                
            # if validation loss better than previous checkpoint, save as best
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

            
            if (current_epoch == self.epoch_lim): 
                print('current epoch = epoch limit, terminating')
            if current_epoch - best_epoch == self.epoch_early_stop:
                print('early stopping')
            
        # Save training progress figure
        utils.plot_training_progress( self.model_path, self.perf_log)
        print('Training Completed')
        
        best_checkpoint_path = os.path.join(self.model_path, 'best_checkpoint.pt') 
        best_model,_,_,_,input_norm_dict  = load_model_and_random_state(best_checkpoint_path, model)
        for dataloader,fold in zip((train_dataloader,val_dataloader,test_dataloader),('train','val','test')):
            self._evaluate(best_model, dataloader, fold, input_norm_dict)

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
        y_true, y_pred, EIDs, loss = run_model_on_dataloader(model, dataloader, self.loss_fn)
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
        train_dl,val_dl,test_dl,input_norm_dict = self._setup_data(input_norm_dict)

        for fold,dataloader in zip((train_dl,val_dl,test_dl),('train','val','test')):
            self._evaluate(model, dataloader, fold, input_norm_dict)
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
        _,_,dataloader,input_norm_dict = self._setup_data(input_norm_dict)
        
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
        _,_,test_dataloader,input_norm_dict = self._setup_data(input_norm_dict)
        
        # Run on test dataset
        print('run model on test set')
        return_model_outputs, return_correct_outputs, return_EIDs, loss = run_model_on_dataloader(best_model, test_dataloader, self.loss_fn)
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

# # WGL: draft of a batch version of this function
# def run_model_on_dataloader(
#     model: torch.nn.Module,
#     dataloader: torch.utils.data.DataLoader,
#     loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
#     *,
#     device: Optional[Union[str, torch.device]] = None,
# ) -> Tuple[torch.Tensor, torch.Tensor, List[int], float]:
#     """
#     Assumes Dataset.__getitem__ returns:
#         embedding, label, eid   (eid is an int)

#     Returns:
#         preds: (N, ...) CPU tensor
#         labels: (N, ...) CPU tensor
#         eids: List[int] length N
#         mean_loss: float (per-sample)
#     """
#     model.eval()

#     if device is None:
#         try:
#             device = next(model.parameters()).device
#         except StopIteration:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     else:
#         device = torch.device(device)

#     preds: List[torch.Tensor] = []
#     labels: List[torch.Tensor] = []
#     eids_all: List[int] = []

#     total_loss = 0.0
#     total_n = 0

#     pbar = tqdm(dataloader, total=len(dataloader.dataset), unit="samples")

#     with torch.inference_mode():
#         for embedding, label, eid in pbar:
#             # embedding: (B, ...)
#             # label:     (B, ...)
#             # eid:       tensor shape (B,) or scalar
#             # print('embedding shape:',embedding.shape)
#             embedding = embedding.to(device, non_blocking=True)
#             label = label.to(device, non_blocking=True)

#             outputs = model(embedding)

#             loss_t = loss_func(outputs, label)
#             B = embedding.shape[0]
#             total_n += B

#             # Correct averaging regardless of loss reduction
#             if loss_t.ndim == 0:
#                 total_loss += float(loss_t) * B
#                 loss_display = float(loss_t)
#             else:
#                 total_loss += float(loss_t.sum())
#                 loss_display = float(loss_t.mean())

#             preds.append(outputs.cpu())
#             labels.append(label.cpu())

#             # ---- EID handling ----
#             if torch.is_tensor(eid):
#                 eids_all.extend(int(v) for v in eid.cpu().tolist())
#             else:
#                 # batch_size == 1 fallback
#                 eids_all.append(int(eid))

#             # pbar.set_postfix(loss=loss_display)
#             # pbar.update(B)
#     # pbar.close()

#     preds = torch.cat(preds, dim=0)
#     labels = torch.cat(labels, dim=0)
#     mean_loss = total_loss / total_n

#     return preds, labels, eids_all, mean_loss

def run_model_on_dataloader(model, dataloader, loss_func_pointer):
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
    for embedding, correct_labels, eid in pbar:
        if torch.cuda.is_available():
            embedding = embedding.to("cuda")
            correct_labels = correct_labels.to("cuda")


        with torch.no_grad():
            model_outputs = model(embedding)
            return_model_outputs.append(model_outputs.to('cpu'))
            return_correct_outputs.append(correct_labels.to('cpu'))
            return_EIDs.append(eid)
            loss += float(loss_func_pointer(model_outputs, correct_labels).to("cpu"))
# 
    return return_model_outputs, return_correct_outputs, return_EIDs, loss


    

if __name__ == "__main__":
    fire.Fire(EchoFocus)
