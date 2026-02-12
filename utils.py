"""Utility functions for normalization, plotting, and reproducibility."""

import os
import math
import numpy as np
import scipy
import torch
import inspect
from functools import wraps
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

def get_norm_params(train_df, measure_list):
    """Compute per-measure normalization parameters for label transforms.

    The transform shifts each measure, optionally applies Box-Cox, then stores
    mean and standard deviation for z-scoring.

    Args:
        train_df: Pandas DataFrame with training labels.
        measure_list: Iterable of column names to normalize.

    Returns:
        dict: Mapping of measure name to normalization parameters.
    """
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
    """Apply saved normalization parameters to a dataframe.

    Args:
        target_df: Pandas DataFrame to normalize.
        input_norm_dict: Dict produced by :func:`get_norm_params`.

    Returns:
        pandas.DataFrame: Normalized copy of ``target_df``.
    """
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
    """Compute sum of squared errors while ignoring NaN targets.

    Args:
        Model_Out (torch.Tensor): Model outputs.
        Correct_Out (torch.Tensor): Target outputs with possible NaNs.

    Returns:
        torch.Tensor: Scalar SSE over non-NaN entries.
    """
    not_nan_mask = ~Correct_Out.isnan().squeeze() # only calculate loss for not-nan-entries
    return torch.nn.functional.mse_loss(Model_Out[not_nan_mask].squeeze(),Correct_Out[not_nan_mask].squeeze())*sum(not_nan_mask) # and we want SSE, not MSE

def un_normalize_output(output, measure_list, input_norm_dict):
    """Invert normalization on model outputs.

    Args:
        output (numpy.ndarray or torch.Tensor): Normalized outputs shaped (N, T) or (T,).
        measure_list: Iterable of measure names in output order.
        input_norm_dict: Dict produced by :func:`get_norm_params`.

    Returns:
        numpy.ndarray or torch.Tensor: Un-normalized outputs with original scale.
    """
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
    Assign ``__init__`` args to instance attributes.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')

    Args:
        func (callable): Initializer to wrap.

    Returns:
        callable: Wrapped initializer that sets attributes before running.
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
    """Restore numpy/torch RNG state from a checkpoint dict.

    Args:
        import_dict (dict): Checkpoint dictionary with RNG state entries.
    """
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
    """Compute the logistic sigmoid of ``z``.

    Args:
        z (numpy.ndarray or float): Input value(s).

    Returns:
        numpy.ndarray or float: Sigmoid-transformed values.
    """
    return 1/(1 + np.exp(-z))

def scatter_plots(model_path, dataset, fold, task_labels, y_true, y_pred):
    """Save scatter plots comparing predicted vs. true labels.

    Args:
        model_path (str): Output directory for plots.
        dataset (str): Dataset name for file labeling.
        fold (str): Split name (e.g., train/val/test).
        task_labels (list[str]): Task label names.
        y_true (numpy.ndarray): Ground-truth values.
        y_pred (numpy.ndarray): Predicted values.
    """
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
    plt.close(fig)


def _make_subplot_grid(num_panels, max_cols=4, panel_width=4, panel_height=3.5):
    """Create a subplot grid and return (fig, flat_axes)."""
    n_cols = min(max_cols, max(1, num_panels))
    n_rows = int(math.ceil(num_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows),
        squeeze=False,
    )
    return fig, axes.flatten()


def plot_roc_curves(model_path, dataset, fold, task_labels, y_true, y_score):
    """Save per-label ROC curves for classification tasks."""
    figure_loc = os.path.join(model_path, f'roc_curves_{fold}_{dataset}.pdf')
    fig, axes = _make_subplot_grid(len(task_labels))
    for i, label in enumerate(task_labels):
        ax = axes[i]
        yt = y_true[:, i]
        ys = y_score[:, i]
        mask = np.isfinite(yt) & np.isfinite(ys)
        yt = yt[mask]
        ys = ys[mask]
        if yt.size == 0 or np.unique(yt).size < 2:
            ax.text(0.5, 0.5, 'Insufficient class\nvariation', ha='center', va='center')
            ax.set_title(label)
            ax.set_axis_off()
            continue
        fpr, tpr, _ = roc_curve(yt, ys)
        auc_val = roc_auc_score(yt, ys)
        ax.plot(fpr, tpr, label=f'AUC={auc_val:.3f}')
        ax.plot([0, 1], [0, 1], '--k', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(label)
        ax.legend(loc='lower right', fontsize=8)
    for j in range(len(task_labels), len(axes)):
        axes[j].set_axis_off()
    plt.tight_layout()
    print('saving',figure_loc)
    fig.savefig(figure_loc, bbox_inches='tight')
    plt.close(fig)


def plot_pr_curves(model_path, dataset, fold, task_labels, y_true, y_score):
    """Save per-label precision-recall curves for classification tasks."""
    figure_loc = os.path.join(model_path, f'pr_curves_{fold}_{dataset}.pdf')
    fig, axes = _make_subplot_grid(len(task_labels))
    for i, label in enumerate(task_labels):
        ax = axes[i]
        yt = y_true[:, i]
        ys = y_score[:, i]
        mask = np.isfinite(yt) & np.isfinite(ys)
        yt = yt[mask]
        ys = ys[mask]
        if yt.size == 0 or np.unique(yt).size < 2:
            ax.text(0.5, 0.5, 'Insufficient class\nvariation', ha='center', va='center')
            ax.set_title(label)
            ax.set_axis_off()
            continue
        precision, recall, _ = precision_recall_curve(yt, ys)
        ap_val = average_precision_score(yt, ys)
        ax.plot(recall, precision, label=f'AP={ap_val:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(label)
        ax.legend(loc='lower left', fontsize=8)
    for j in range(len(task_labels), len(axes)):
        axes[j].set_axis_off()
    plt.tight_layout()
    print('saving',figure_loc)
    fig.savefig(figure_loc, bbox_inches='tight')
    plt.close(fig)

def plot_training_progress(model_path, perf_log):
    """Save a training/validation loss plot.

    Args:
        model_path (str): Output directory for the plot.
        perf_log (list[list]): Rows of [epoch, train_loss, val_loss, lr, epoch_time].
    """
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
