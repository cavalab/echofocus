import os
import numpy as np
import scipy
import torch
import inspect
from functools import wraps
import matplotlib.pyplot as plt

def get_norm_params(train_df, measure_list):
    input_norm_dict = {}
    input_norm_dict['type'] = 'boxcox'
    
    # per measure
    # 1) shift. I want the "min" to be one "abs(mean)" above 0
    # 2) boxcox power norm
    # 3) subtract mean
    # 4) divide by stdev
    
    if (train_df is not None):
        for k in measure_list:
            tmp_dict = {} # store the lambda, mean, and stdev for the measure
            tmp = train_df[train_df[k].notna()][k].values # use non-zero inputs

            shift = np.abs(np.mean(tmp)) - np.min(tmp) # set min to zero, then increase by abs(mean)
            tmp_dict['shift'] = shift
            
            tmp = tmp + shift
            # for this modeling variant, only apply to measures in [0,500] range
            if (k in ['LD05','LM12','LS04','LA34','RA06']):
                tmp_dict['apply_boxcox'] = True 
                standardized_tmp, lmbda = scipy.stats.boxcox(tmp) # standardize after shifting and get lambda param
                tmp_dict['lmbda'] = lmbda
            else:
                tmp_dict['apply_boxcox'] = False
                standardized_tmp = tmp
            
            # then we need the mean and stdev to z-score after making the data look normal
            tmp_dict['mean'] = np.mean(standardized_tmp)
            tmp_dict['stdev'] = np.std(standardized_tmp)
            
            input_norm_dict[k] = tmp_dict
    else:
       print('\nno train_df! cant generate norm! error?\n')
       
    return input_norm_dict

def normalize_df(target_df, input_norm_dict):
    # requires input_norm_dict from get_norm_params
    out_df = target_df.copy() # I think this is the preferred way to change a dataframe?
    keys = [key for key in input_norm_dict.keys()]
    for measure in keys:
        if measure != 'type': # "type" is a key in input_norm_dict indicating the type of transform - not relevant here, so skip it
            tmp_dict = input_norm_dict[measure] # get the dictionary with shift, lmbda, mean, stdev for the measure
            values = target_df[measure].values # pull values
            values += tmp_dict['shift'] # shift
            
            if tmp_dict['apply_boxcox']:
                values = scipy.stats.boxcox(values, lmbda = tmp_dict['lmbda']) # transform
                
            values = values - tmp_dict['mean'] # mean of 0
            values = values / tmp_dict['stdev'] # stdev of 1
            out_df[measure] = values
    return out_df 
def masked_mse_loss(Model_Out, Correct_Out): # SSE
        not_nan_mask = ~Correct_Out.isnan().squeeze() # only calculate loss for not-nan-entries
        return torch.nn.functional.mse_loss(Model_Out[not_nan_mask].squeeze(),Correct_Out[not_nan_mask].squeeze())*sum(not_nan_mask) # and we want SSE, not MSE

def un_normalize_output(output, measure_list, input_norm_dict):
    if len(output.shape) == 1:
        output = output.reshape(1,-1)
    assert output.shape[1] == len(measure_list)
    # if len(output.shape)>1:
    #     if output.shape[1]==1:
    #         output=output.reshape(-1)
    for i,key in enumerate(measure_list):
        tmp_dict = input_norm_dict[key]
        shift = tmp_dict['shift']
        
        apply_boxcox = tmp_dict['apply_boxcox']
        
        mean = tmp_dict['mean']
        stdev = tmp_dict['stdev']

        tmp = output[:,i] # get post-normalization values
        tmp  = (tmp  * stdev) + mean # invert z-score
        
        if apply_boxcox:
            lmbda = tmp_dict['lmbda']
            tmp  = np.array([ scipy.special.inv_boxcox(k , lmbda) for k in tmp]) # invert box-cox
        tmp = tmp - shift # invert data shift
        output[:,i] = torch.from_numpy(tmp) # return
        
        # tmp = correct_output[:,i] # get post-normalization values
        # tmp  = (tmp  * stdev) + mean # invert z-score
        # if apply_boxcox:
        #     lmbda = tmp_dict['lmbda']
        #     tmp  = np.array([ scipy.special.inv_boxcox(k , lmbda) for k in tmp]) # invert box-cox
        # tmp = tmp - shift # invert data shift
        # correct_output[:,i] = tmp # return
    
    return output


def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    # names, varargs, keywords, defaults = inspect.getargspec(func)
    names, varargs, keywords, defaults, _,_,_ = inspect.getfullargspec(func)
    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper

def load_random_state(import_dict):
    if ( ('Numpy_Random_State' in import_dict.keys()) and ('Torch_Random_State' in import_dict.keys()) ):
        np.random.set_state(import_dict['Numpy_Random_State'])
        torch.random.set_rng_state(import_dict['Torch_Random_State'])
        if ('CUDA_Random_State' in import_dict.keys()):
            torch.cuda.random.set_rng_state(import_dict['CUDA_Random_State'])
        else:
            print('CUDA RANDOM STATE NOT LOADED. Further training not deterministic')
        torch.backends.cudnn.deterministic = True # make TRUE if you want reproducible results (slower)
    else:
        print('Could not load random state')

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def scatter_plots(model_path, dataset, fold, task_labels, y_true, y_pred):
    # draw scatterplot with 1:1 dashed line on diagonal
    figure_loc = os.path.join(model_path, f'scatter_plots_{fold}_{dataset}.pdf')
    fig, ax = plt.subplots(3,6,figsize=(20,10))
    for i,k in enumerate(task_labels):
        row = int(i/6)
        col = i-6*row
        ax[row, col].scatter(y_true[:,i], y_pred[:,i],alpha=0.15)
        # find where to draw the line...
        all_values = np.concatenate([y_true[:,i], y_pred[:,i]]) 
        lim_loc = np.argmax(np.abs(all_values[~np.isnan(all_values)]))
        lim = all_values[~np.isnan(all_values)][lim_loc]
        ax[row, col].plot([0,lim], [0,lim],'--k')
        ax[row, col].set_xlabel('Correct '+k)
        ax[row, col].set_ylabel('Predict '+k)
        ax[row, col].set_title(k)
    plt.tight_layout()
    print('saving',figure_loc)
    fig.savefig(figure_loc, bbox_inches = 'tight')

def plot_training_progress(model_path, perf_log):
    figure_loc = os.path.join(model_path, 'training_progress.pdf')
    f = plt.figure() # https://stackoverflow.com/questions/11328958/save-multiple-plots-in-a-single-pdf-file
    epochs      = [k[0] for k in perf_log]
    train_perf  = [k[1] for k in perf_log]
    val_perf    = [k[2] for k in perf_log]
    plt.scatter(epochs, train_perf)
    plt.scatter(epochs, val_perf)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train Loss','Val Loss'])
    plt.title('Training Progress')
    print('saving',figure_loc)
    f.savefig(figure_loc, bbox_inches = 'tight')