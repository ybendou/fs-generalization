import random
import torch
import numpy as np
import random
import scipy.stats as st
import pickle 

def confInterval(scores):
    if scores.shape[0] == 1:
        low, up = -1., -1.
    elif scores.shape[0] < 30:
        low, up = st.t.interval(0.95, df = scores.shape[0] - 1, loc = scores.mean(), scale = st.sem(scores.numpy()))
    else:
        low, up = st.norm.interval(0.95, loc = scores.mean(), scale = st.sem(scores.numpy()))
    return low, up

def fix_seed(seed, deterministic=False):
    ### generate random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def stats(scores, name):
    if len(scores) == 1:
        low, up = 0., 1.
    elif len(scores) < 30:
        low, up = st.t.interval(0.95, df = len(scores) - 1, loc = np.mean(scores), scale = st.sem(scores))
    else:
        low, up = st.norm.interval(0.95, loc = np.mean(scores), scale = st.sem(scores))
    if name == "":
        return np.mean(scores), up - np.mean(scores)
    else:
        #print("{:s} {:.2f} (± {:.2f}) (conf: [{:.2f}, {:.2f}]) (worst: {:.2f}, best: {:.2f})".format(name, 100 * np.mean(scores), 100 * np.std(scores), 100 * low, 100 * up, 100 * np.min(scores), 100 * np.max(scores)))
        print("{:s} {:.2f}% ± {:.2f}% (conf: [{:.2f}, {:.2f}]) (worst: {:.2f}, best: {:.2f})".format(name, 100 * np.mean(scores), 100 * (up-low), 100 * low, 100 * up, 100 * np.min(scores), 100 * np.max(scores)))

def sphering(features):
    return features / torch.norm(features, p = 2, dim = 2, keepdim = True)

def sphering_L1(features):
    return features / torch.norm(features, p = 3, dim = 2, keepdim = True)

def centering(train_features, features, base_mean=False):
    if base_mean:
        return features - train_features
    else:
        return features - train_features.reshape(-1, train_features.shape[2]).mean(dim = 0).unsqueeze(0).unsqueeze(0)

def preprocess(train_features, features, preprocessing, base_mean=False):
    """
        Preprocess data.
        - base_mean : if True then train_features is already averaged.
    """
    for i in range(len(preprocessing)):
        if preprocessing[i] == 'R':
            if not base_mean:
                with torch.no_grad():
                    train_features = torch.relu(train_features)
            features = torch.relu(features)
        if preprocessing[i] == 'P':
            if not base_mean:
                with torch.no_grad():
                    train_features = torch.pow(train_features, 0.5)
            features = torch.pow(features, 0.5)
        if preprocessing[i] == 'E':
            if not base_mean:
                with torch.no_grad():
                    train_features = sphering(train_features)
            features = sphering(features)
        if preprocessing[i] == 'M':
            features = centering(train_features, features, base_mean)
            if not base_mean:
                with torch.no_grad():
                    train_features = centering(train_features, train_features)
    return features

def postprocess(runs, args, train_features=None):
    # runs shape: [100, 5, 16, 640]
    for i in range(len(args.postprocessing)):
        if args.postprocessing[i] == 'R':
            runs = torch.relu(runs)
        if args.postprocessing[i] == 'P':
            runs = torch.pow(runs, 0.5)
        if args.postprocessing[i] == 'E':
            runs = runs/torch.norm(runs, p=2, dim=3, keepdim=True)
        if args.postprocessing[i] == 'N': # substract mean of novel data instead of base (transductive setting)
            runs = runs - runs.reshape(runs.shape[0], -1, runs.shape[-1]).mean(dim=1, keepdim=True).unsqueeze(1)
        if args.postprocessing[i] == 'M' and train_features!=None:
            runs = runs - train_features.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return runs


def dimReduction(means):
    """
        Performs QR reduction on axis of centroids and return the projection matrix Q.
    """
    perm = torch.arange(means.shape[1])-1
    LDAdirections = (means-means[:,perm])[:, :-1].squeeze(0)
    Q, R = torch.linalg.qr(LDAdirections.T)
    return Q.T

def load_features(features_path, base_features_path='', device='cpu', return_mean_base=False):
    """
    Load features and concatenate them given a list of features.
    """

    if features_path!=base_features_path:
        novel_features_list = []
        if type(features_path)==str: 
            features_path = [features_path]
        for file_path in features_path: # stack crops
            feats = torch.load(file_path, map_location='cpu')
            if type(feats)==dict:
                feats = feats['augmented']
            else:
                if feats.shape[0]==100: feats = feats[80:] # if features include base, val and novel
            novel_features_list.append(feats.reshape(20, 600, -1, feats.shape[-1]))
            
        novel_features = torch.cat(novel_features_list, dim=2)
        del feats, novel_features_list
        
        AS_feats = novel_features.mean(dim=2).to(device) # get average of features

        # Get features of the base dataset 
        if base_features_path!='':
            print('V4')
            feats = torch.load(base_features_path, map_location=device)
            if 'mean' in base_features_path: # if loading directly the mean vector of the base classes
                return novel_features, AS_feats, feats
            else:
                base_features = feats[:64]
                if return_mean_base:
                    base_features = torch.mean(base_features.reshape(-1, base_features.shape[-1]), dim=0).to('cpu')
                return novel_features, AS_feats, base_features
        return novel_features, AS_feats  
    else:
        feats = torch.load(base_features_path, map_location=device)
        base_features = feats[:64]
        AS_feats = feats[80:]
        novel_features = AS_feats.unsqueeze(2)
        
        return novel_features, AS_feats, base_features
    
def fastpickledump(obj, file):
    """
        Dump object to pickle file.
    """
    with open(file, 'wb') as f:
        p = pickle.Pickler(f)
        p.fast = True
        p.dump(obj)
        
def unravel_index(
    indices,
    shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord