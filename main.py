import numpy as np
from itertools import combinations
import argparse
from scipy import interpolate
import pandas as pd 
import os
from tqdm import tqdm 
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from utils import stats
from few_shot_evaluation import EpisodicGenerator
from args import process_arguments

def generate_normal_data(means, covariance, N=1000, use_multivariate=False):
    """
    Generate random samples for n-way gaussians
    """
    data = torch.zeros(means.shape[0], N, means.shape[-1])
    for i, mu in enumerate(means):
        cov = covariance[i]
        if use_multivariate:
            data[i] = torch.tensor(np.random.multivariate_normal(mu.cpu(), cov.cpu(), N))
        else:
            data[i] = torch.randn(N, means.shape[-1])@cov+mu
    return data
def return_error(data, means):
    """

    """
    score, total = 0., 0.
    for c, queries in enumerate(data):
        distances = torch.cdist(queries, means, p=2)
        winners = distances.argmin(dim=1)
        score += (winners==c).sum()
        total += len(queries)
    return 1-score/total
def predict_error(means=None, covariance=None, N=10000, use_multivariate=False):
    """
    Predict accuracy of the model
    """
    data = generate_normal_data(means, covariance, N=N, use_multivariate=use_multivariate)        
    score = return_error(data, means)
    return score

def dimReduction(means):
    """
        Performs QR reduction on axis of centroids and return the projection matrix Q.
        returns : 
        - Q: the projection matrix
        - direction_coordinates : the pairs of classes used to compute the directions of QR directions
    """
    perm = torch.arange(means.shape[1])-1
    c = means.shape[1]
    direction_coordiantes = torch.stack([torch.arange(c), torch.arange(c)[torch.arange(c)-1]]).T[:-1]
    LDAdirections = (means-means[:,perm])[:, :-1].squeeze(0)
    Q, R = torch.linalg.qr(LDAdirections.T)
    return Q.T, direction_coordiantes

# find n points such that the distance between the n points is the new distance 
def get_k_shot_points(k, mean, cov):
    return [torch.randn(k,N_dims)@cov[c] + mean[c] for c in range(len(mean))],[[c]*k for c in range(len(mean))]

def get_value_upper(M):
    return torch.cat([torch.stack([M[c2, c1] for c2 in range(c1)]) for c1 in range(1, len(M))])

def interpolate_bias(estimated_snr, K, mapping):
    if estimated_snr<mapping[K].x.min():
        return 0 
    elif estimated_snr>mapping[K].x.max():
        return 1
    else:
        return mapping[K](estimated_snr)

def cycle_list(lst, n):
    """
    splits a list into n parts and cycles through them
    """
    extra_lst = lst + lst[:len(lst)//n]
    return [extra_lst[i:i+n] for i in range(len(lst))], [lst[max(0,i-len(lst)+n):i]+lst[i+n:len(lst)+i] for i in range(len(lst))]#[lst[len(lst)//(i+1)-1:len(lst)//(i+1)]+lst[i+n:] for i in range(len(lst))]

def split_val(episode, n_val, choice='shots', kfold=False):
    """
    Add validation set to either shots or queries
    """
    assert choice in ['shots', 'queries'], 'choice is either shots or queries'
    
    if kfold:
        # create multiple episodes with different validation sets
        n_episodes = [len(episode[f'{choice}_idx'][c]) for c in range(len(episode[f'{choice}_idx']))]#[int(int(len(episode[f'{choice}_idx'][c])/n_val)) for c in range(len(episode[f'{choice}_idx']))]
        n_episodes = min(n_episodes) # get the minimum number of folds for all classes in case it's unbalanced shots
        reduced_episode = []
        folds = [cycle_list(indices, n_val) for indices in episode[f'{choice}_idx']] # split the indices into n_val folds for each class, each fold moves one index at a time 
        for i in range(n_episodes):
            new_episode = episode.copy()
            new_episode['validations_idx'] = [folds[c][0][i] for c in range(len(folds))]
            new_episode[f'{choice}_idx'] = [folds[c][1][i] for c in range(len(folds))]
            # new_episode['validations_idx'] = [indices[i*n_val:(i+1)*n_val] for indices in new_episode[f'{choice}_idx']]       
            # new_episode[f'{choice}_idx'] = [indices[:i*n_val]+indices[(i+1)*n_val:] for indices in new_episode[f'{choice}_idx']]
            reduced_episode.append(new_episode)
        return reduced_episode
    else:
        reduced_episode = episode.copy()
        reduced_episode['validations_idx'] = [indices[:n_val] for indices in reduced_episode[f'{choice}_idx']]
        reduced_episode[f'{choice}_idx'] = [indices[n_val:] for indices in reduced_episode[f'{choice}_idx']]
        return reduced_episode
    
def gradient_descent_ClosedForm(points, target_distances, trainCfg={'lr':0.1, 'mmt':0.8, 'n_iter':100, 'loss_amp':1}, device='cuda:0', verbose=False):
    """
    """
    # load images as tensors
    points = points.to(device) 
    target_distances = target_distances.to(device)
    L2 = nn.MSELoss()

    solution = nn.Parameter(points.clone().to(device))
    optimizer = torch.optim.SGD([solution], lr=trainCfg['lr'], momentum=trainCfg['mmt'])
    best_iter = {'loss':10e5, 'n':0, 'loss_reg':10e5, 'points':solution.data.cpu().clone()}
    count_val = 0
    for n in range(trainCfg['n_iter']):
        
        optimizer.zero_grad()
        
        distances = torch.cdist(solution, solution, p=2).triu()
        distances = get_value_upper(distances)
        lossMSE = L2(distances, target_distances)
        lossMSE.backward()
        optimizer.step()

        with torch.no_grad():
            if count_val%10==0 and verbose:
                print(f'Iter {n} | Loss: {lossMSE.item()}')
            count_val += 1
            if lossMSE.item()<= best_iter['loss']:
                best_iter['loss'] = lossMSE.item()
                best_iter['points'] = solution.data.cpu().clone()
                best_iter['n'] = n
    if verbose:
        print(f'best epoch Iter {best_iter["n"]} | Loss: {best_iter["loss"]}')

    return best_iter['points']   # load images as tensors
def estimate_bias(n_runs , N_dims, bessel_correction, maxK):
    d = 5
    center = torch.ones(N_dims)*d/np.sqrt(N_dims)
    iterator = list(np.linspace(0.01, 6, 100))
    Ks = list(range(1, maxK+1, 1))
    maxK = len(Ks)
    measure_estimate = torch.zeros(maxK, len(iterator),n_runs, 3)
    for k, K in tqdm(enumerate(Ks)):
        for s, sigma_val in enumerate(iterator):
            covariances = [torch.eye(N_dims)*sigma_val]
            for i in range(n_runs):
                k_shot_points, k_shot_labels = get_k_shot_points(K, center.unsqueeze(0), covariances)
                # measure statistics
                estimated_centroid = k_shot_points[0].mean(dim=0)
                if K>1:
                    estimated_sigma = k_shot_points[0].flatten().T.cov(correction=bessel_correction).sqrt() 
                else:
                    estimated_sigma = 1
                Q, _ = dimReduction(torch.stack([estimated_centroid, torch.zeros(N_dims)]).unsqueeze(0))
                qr_estimated_centroid = torch.einsum('nd,d->n', Q, estimated_centroid)
                qr_center = torch.einsum('nd,d->n', Q, center)
                d_estimate = (qr_center/qr_estimated_centroid).abs()
                measure_estimate[k, s, i, 0] = d_estimate
                measure_estimate[k, s, i, 1] = qr_estimated_centroid.abs()/estimated_sigma
                measure_estimate[k, s, i, 2] = qr_center.abs()/estimated_sigma     
    mapping = {K:interpolate.interp1d(measure_estimate.mean(dim=2)[k][:, 1].numpy(), measure_estimate.mean(dim=2)[k][:, 0].numpy(), kind = 'cubic') for k,K in enumerate(Ks)}
    return mapping

def get_features_from_indices(features, episode, validation=False):
    """
    Get features from a list of all features and from a dictonnary describing an episode
    """
    choice_classes, shots_idx, queries_idx = episode['choice_classes'], episode['shots_idx'], episode['queries_idx']
    if validation : 
        validation_idx = episode['validations_idx']
        val = []
    shots, queries = [], []
    for i, c in enumerate(choice_classes):
        shots.append(features[c]['features'][shots_idx[i]])
        queries.append(features[c]['features'][queries_idx[i]])
        if validation : 
            val.append(features[c]['features'][validation_idx[i]])
    if validation:
        return shots, queries, val
    else:
        return shots, queries
def DavisBouldinIndex(centroids, clusters):
    K = len(clusters)
    M = torch.cdist(centroids, centroids, p=2)
    S = torch.stack([(clusters[c]-centroids[c]).pow(2).mean().sqrt() for c in range(len(clusters))])
    C = torch.stack([max([(S[i]+ S[j]/M[i,j]) for j in range(K) if j!=i]) for i in range(K)]).mean()
    return C

def main(n_runs, c, maxK, N_dims, bessel_correction, generator, features, mapping, unbalanced_queries=False):
    measure = torch.zeros(n_runs, maxK, 12)
    min_size = min([features[c]['features'].shape[0] for c in range(len(features))])
    for i in tqdm(range(n_runs)):
        episode = generator.sample_episode(ways=c, n_shots=1, n_queries=min_size, unbalanced_queries=unbalanced_queries)
        _, queries = generator.get_features_from_indices(features, episode)
        for k, K in enumerate(range(1, maxK+1)):
            k_shot_points = [queries[c_][:K] for c_ in range(c)]
            query_points = [queries[c_][maxK+1:] for c_ in range(c)]
            # measure statistics
            estimated_centroids = torch.stack([shot.mean(dim=0) for shot in k_shot_points])
            true_error = return_error(query_points, estimated_centroids)
            if K>1:
                estimated_sigmas = torch.stack([shot.T.cov(correction=bessel_correction).diag().mean().sqrt() for shot in k_shot_points])
            else:
                estimated_sigmas = torch.ones(c)
            centroids_queries = torch.stack([q.mean(dim=0) for q in query_points])

            Q, _ = dimReduction(estimated_centroids.unsqueeze(0))
            qr_estimated_centroids = torch.einsum('nd,cd->cn', Q, estimated_centroids)    # Project centroids in the QR space
            qr_k_shot_points = [torch.einsum('nd,kd->kn', Q, shots) for shots in k_shot_points] # Project shots in the QR space

            Q_true_centroids, _ = dimReduction(centroids_queries.unsqueeze(0)) # QR projection with true centroids 
            qr_queries = [torch.einsum('nd,kd->kn', Q_true_centroids, q) for q in query_points] # queries projected in the QR space spanned by the true centroids
            all_qr_shots_cov = torch.cat([qr_k_shot_points[c_]-qr_k_shot_points[c_].mean(dim=0) for c_ in range(c)]).T.cov(correction=bessel_correction) 

            if c>2:
                qr_sigmas_queries = torch.stack([q.T.cov(correction=bessel_correction).diag().mean().sqrt() for q in qr_queries])
                qr_estimated_shared_sigmas = torch.ones(c)*all_qr_shots_cov.diag().mean().sqrt()
                if K>1:
                    qr_estimated_sigmas = torch.stack([shot.T.cov(correction=bessel_correction).diag().mean().sqrt() for shot in qr_k_shot_points]) 
                else:
                    qr_estimated_sigmas = torch.ones(c)
            else:
                if K>1:
                    qr_estimated_sigmas = torch.stack([shot.T.cov(correction=bessel_correction).mean().sqrt() for shot in qr_k_shot_points])
                else:
                    qr_estimated_sigmas = torch.ones(c)
                qr_sigmas_queries = torch.stack([q.T.cov(correction=bessel_correction).mean().sqrt() for q in qr_queries])
                qr_estimated_shared_sigmas = torch.ones(c)*all_qr_shots_cov.mean().sqrt()
        

            qr_centroids_queries = torch.stack([q.mean(dim=0) for q in qr_queries])
            qr_estimated_covariances = [torch.eye(c-1)*qr_estimated_sigmas[c_] for c_ in range(c)]
            qr_estimated_shared_covariances = [torch.eye(c-1)*qr_estimated_shared_sigmas[c_] for c_ in range(c)]        

            qr_covariances_queries = [torch.eye(c-1)*qr_sigmas_queries[c_] for c_ in range(c)]        
            
            # Correct distance given the measured SNRs (estiamted_distance/estimated_sigma)
            distances = torch.cdist(qr_estimated_centroids, qr_estimated_centroids).triu() # First estimate of the distances
            corrected_distances = torch.zeros(c, c)
            for (c1, c2) in torch.cat([torch.stack([torch.tensor([c2, c1]) for c2 in range(c1)]) for c1 in range(1, c)]):
                estimated_distance = distances[c1, c2]
                estimated_SNR = estimated_distance/(estimated_sigmas[c1]**2+estimated_sigmas[c2]**2).sqrt()
                adjusting_ratio = torch.tensor(interpolate_bias(estimated_SNR.numpy(), K, mapping)).abs()
                corrected_distance = estimated_distance*adjusting_ratio
                corrected_distances[c1, c2] = corrected_distance

            target_distances = get_value_upper(corrected_distances)
            corrected_centroids = gradient_descent_ClosedForm(qr_estimated_centroids, torch.clip(target_distances, 0), verbose=False, trainCfg={'lr':0.01, 'mmt':0.8, 'n_iter':100, 'loss_amp':1}) # Retrieve new centroids 
            
            measure[i][k][0] = true_error # true acc in N_dims 
            measure[i][k][1] = return_error(qr_queries, qr_estimated_centroids) # true acc in QR space
            measure[i][k][2] = predict_error(qr_centroids_queries, qr_covariances_queries) # If we know all the queries in QR
            measure[i][k][3] = predict_error(corrected_centroids, qr_estimated_shared_covariances) # If we use shared covariance instead
            measure[i][k][4] = DavisBouldinIndex(estimated_centroids, k_shot_points) # DB index in high dimensions
            measure[i][k][5] = DavisBouldinIndex(qr_estimated_centroids, qr_k_shot_points) # DB index in QR space
            measure[i][k][6] = predict_error(corrected_centroids, qr_estimated_covariances)  # If we use the corrected centroid
            measure[i][k][7] = predict_error(corrected_centroids, qr_covariances_queries)  # If we use the corrected centroid and the true sigmas 
            measure[i][k][9] = predict_error(corrected_centroids, [torch.eye(c-1) for _ in range(c)])  # If we use the Identity matrix as covariance
            
            if K>c-1:
                measure[i][k][10] = predict_error(corrected_centroids, [shot.T.cov(correction=bessel_correction) for shot in qr_k_shot_points], use_multivariate=True)  # If we use the entire covariance matrix
            measure[i][k][11] = predict_error(qr_estimated_centroids, qr_estimated_shared_covariances)  # Our method without bias correction

            # Cross-validation model
            new_episode = episode.copy()
            new_episode['shots_idx'] = [q[:K] for q in new_episode['queries_idx']]
            new_episode['queries_idx'] = [q[K:] for q in new_episode['queries_idx']]

            reduced_episode = split_val(new_episode, 1, choice='shots', kfold=True)  # get validation from the queries
            n_episodes = len(reduced_episode)
            val_score = torch.zeros(n_episodes)
            for j in range(n_episodes):
                reduced_shots, _, validations = generator.get_features_from_indices(features, reduced_episode[j], validation=True)
                reduced_centroids = torch.stack([shotClass.mean(dim=0) for shotClass in reduced_shots])
                val_score[j] = return_error(torch.stack(validations), reduced_centroids) # get performance on the validation set
            measure[i][k][8] = val_score.mean()
    return measure
    
from scipy.optimize import curve_fit
# define the true objective function
def adjusted_R_squared(gt, preds, dim):
    R2 = R_squared(gt, preds)
    return 1 - (1-R2)*(len(gt)-1)*(len(gt)-1)/(len(gt)-dim-1), R2
def R_squared(gt, preds):
    return 1-((preds-gt)**2).sum()/((gt-gt.mean())**2).sum()
def objective(x, a, b):
    return a * x + b
def r_square(gt, pred, verbose=False):
    popt, _ = curve_fit(objective, pred, gt)
    # summarize the parameter values
    a, b = popt
    if verbose:
        print('y = %.5f * x + %.5f' % (a, b))
    y = a*pred+b
    return a, b
def beautiful_score(score):
    mean, conf = stats(score, "")
    return '{:.2f}% ± {:.2f}%'.format(100*mean, 100*conf)

def convert_score_to_float(x):
    return x.str[:4].apply(float).values


def MAPE(gt, pred):
    return abs(gt-pred)/(1-gt)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", type=str, default='', help="Report to wandb, input is the entity name")
    parser.add_argument("--wandbProjectName", type=str, default='few-shot', help="wandb project name")
    parser.add_argument("--save-folder", type=str, default='', help="Folder where to save data")
    parser.add_argument("--unbalanced", action="store_true", help="unbalanced dataset")
    parser.add_argument("--validation", action="store_true", help="use this dataset for validation")
    parser.add_argument("--config-validation", type=str, default='', help="Load validation configuration")
    parser.add_argument("--maxK", type=int, default=20, help="max number of shots in the few shot runs")
    parser.add_argument("--load-config", type=str, default='', help="Load pre-computed configuration")

    args = process_arguments(parser=parser)
    if args.wandb:
        import wandb
    dataset = args.dataset
    c = args.n_ways # classes
    n_runs = args.n_runs
    unbalanced = args.unbalanced
    features = torch.load(f'{args.features_path}')
    min_size = min([features[c]['features'].shape[0] for c in range(len(features))])
    maxK = min(int(min_size/2), args.maxK)

    generator = EpisodicGenerator(num_elements_per_class= [len(feat['features']) for feat in features])
    N_dims = features[0]['features'].shape[-1] # dimensions
    success = False
    max_trials = 10
    if not args.load_config:
        while not success:
            try:
                mapping = estimate_bias(n_runs=100 , N_dims=N_dims, bessel_correction=1, maxK=maxK)
                success = True
            except Exception as e:
                max_trials -= 1
                if max_trials == 0:
                    raise ValueError('Could not estimate the bias mapping')
                print(f'Error in the bias estimation: {e}, retrying {max_trials} times')

    bessel_correction = 1
    save_config = f'nruns{n_runs}_c{c}_unbalanced{unbalanced}'
    # ImageNet 
    if args.wandb!='':
            run_wandb = wandb.init(reinit = True, project=args.wandbProjectName, 
            entity=args.wandb, 
            config={'dataset':dataset, 'c':c, 'unbalanced':unbalanced, 'n_runs':n_runs, 'features_path':args.features_path})
    if not args.validation:
        measure_validation = torch.load(args.config_validation)#torch.load(f'{args.save_folder}/metadataset_imagenet_test/{save_config}_filename_{args.features_path_metadataset_imagenet.split("/")[-1].replace(".pt", "")}.pt')

    if args.load_config:
        measure = torch.load(args.load_config)
    else:
        measure = main(n_runs, c, maxK, N_dims, bessel_correction, generator, features, mapping, unbalanced_queries=unbalanced)
    if args.validation:
        measure_validation = measure
        
    if args.save_folder:
        if not os.path.exists(f'{args.save_folder}/{dataset}'):
            os.makedirs(f'{args.save_folder}/{dataset}')
        torch.save(measure, os.path.join(args.save_folder, dataset, f'{save_config}_filename_{args.features_path.split("/")[-1].replace(".pt", "")}.pt'))

    df = pd.DataFrame(columns=['dataset', 'c', 'K', 'unbalanced', 'n_runs', 'oracle', 'ours with true sigmas',  'ours with free sigmas',  'cross-validation',  'oracle unbiased',  'ours unbiased',  'ours with true sigmas unbiased',  'cross-validation unbiased', 'ours without correction', 'ours without correction unbiased'])

    for k,K in enumerate(range(1, maxK+1)):
        config = {'dataset':dataset, 'c':c, 'K':K, 'unbalanced':unbalanced, 'n_runs':n_runs}
        config['oracle'] = beautiful_score(MAPE(measure[:, k, 0], measure[:, k, 2]).tolist())
        config['ours with free sigmas'] = beautiful_score(MAPE(measure[:,k,0], measure[:,k,6]).tolist()) 
        config['ours with shared sigmas'] = beautiful_score(MAPE(measure[:,k,0], measure[:,k,3]).tolist())
        config['ours with true sigmas'] = beautiful_score(MAPE(measure[:,k,0], measure[:,k,7]).tolist())
        config['ours with identity'] = beautiful_score(MAPE(measure[:,k,0], measure[:,k,9]).tolist())
        config['ours with free cov'] = beautiful_score(MAPE(measure[:,k,0], measure[:,k,10]).tolist())
        config['cross-validation'] = beautiful_score(MAPE(measure[:,k,0], measure[:,k,8]).tolist())
        config['ours without correction'] = beautiful_score(MAPE(measure[:,k,0], measure[:,k,11]).tolist())
    
        fits = {}
        a, b = r_square(measure_validation[:, k, 0].numpy(), measure_validation[:, k, 4].numpy())
        fits['DBIndex high'] = a*measure[:, k, 4]+b
        a, b = r_square(measure_validation[:, k, 0].numpy(), measure_validation[:, k, 5].numpy())
        fits['DBIndex low'] = a*measure[:, k, 5]+b

        bias = {'oracle':measure_validation[:, k, 2].mean()-measure_validation[:, k, 0].mean(), 
            'ours with free sigmas':measure_validation[:, k, 6].mean()-measure_validation[:, k, 0].mean(), 
                'ours with true sigmas':measure_validation[:, k, 7].mean()-measure_validation[:, k, 0].mean(), 
                'ours with shared sigmas':measure_validation[:, k, 3].mean()-measure_validation[:, k, 0].mean(), 
                'cross-validation': measure_validation[:, k, 8].mean()-measure_validation[:, k, 0].mean(), 
                'ours with identity':measure_validation[:, k, 9].mean()-measure_validation[:, k, 0].mean(),
                'ours with free cov':measure_validation[:, k, 10].mean()-measure_validation[:, k, 0].mean(),
                'ours without correction':measure_validation[:, k, 11].mean()-measure_validation[:, k, 0].mean()}


        config['oracle unbiased'] = beautiful_score(MAPE(measure[:, k, 0], measure[:, k, 2]-bias['oracle']).tolist())
        config['ours with free sigmas unbiased'] = beautiful_score(MAPE(measure[:, k, 0], measure[:, k, 6]-bias['ours with free sigmas']).tolist())
        config['ours with shared sigmas unbiased'] = beautiful_score(MAPE(measure[:, k, 0], measure[:, k, 3]-bias['ours with shared sigmas']).tolist())
        config['ours with true sigmas unbiased'] = beautiful_score(MAPE(measure[:, k, 0], measure[:, k, 7]-bias['ours with true sigmas']).tolist())
        config['cross-validation unbiased'] = beautiful_score(MAPE(measure[:, k, 0], measure[:, k, 8]-bias['cross-validation']).tolist())
        config['ours with identity unbiased'] = beautiful_score(MAPE(measure[:, k, 0], measure[:, k, 9]-bias['ours with identity']).tolist())
        config['ours with free cov unbiased'] = beautiful_score(MAPE(measure[:, k, 0], measure[:, k, 10]-bias['ours with free cov']).tolist())
        config['ours without correction unbiased'] = beautiful_score(MAPE(measure[:, k, 0], measure[:, k, 11]-bias['ours without correction']).tolist())

        config['DBIndex high'] = beautiful_score(MAPE(measure[:, k, 0], fits['DBIndex high']).tolist())
        config['DBIndex low'] = beautiful_score(MAPE(measure[:, k, 0], fits['DBIndex low']).tolist())

        df = df.append([config])

    if args.save_folder:
        df.to_csv(os.path.join(args.save_folder, dataset, f'{save_config}_filename_{args.features_path.split("/")[-1].replace(".pt", "")}.csv'))
    print(df[['dataset', 'K', 'oracle unbiased', 'ours with identity unbiased', 'ours with free sigmas unbiased',  'ours with shared sigmas unbiased',  'ours with free cov unbiased', 'cross-validation unbiased', 'DBIndex high', 'DBIndex low']])
