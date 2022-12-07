import random
import argparse
import random

def process_arguments(parser=None, params=None):
    if parser == None:
        parser = argparse.ArgumentParser()

    ### pytorch options
    parser.add_argument("--device", type=str, default="cuda:0", help="device(s) to use, for multiple GPUs try cuda:ijk, will not work with 10+ GPUs")
    parser.add_argument("--dataset-path", type=str, default='test/test/', help="dataset path")
    parser.add_argument("--features-path", type=str, default='', help="features directory path")
    
    parser.add_argument("--dataset-device", type=str, default="", help="use a different device for storing the datasets (use 'cpu' if you are lacking VRAM)")
    parser.add_argument("--deterministic", action="store_true", help="use desterministic randomness for reproducibility")

    ### run options
    parser.add_argument("--dataset", type=str, default="miniimagenet", help="dataset to use")
    parser.add_argument("--seed", type=int, default=-1, help="set random seed manually, and also use deterministic approach")

    ### few-shot parameters
    parser.add_argument("--n-shots", type=str, default="[1,5]", help="how many shots per few-shot run, can be int or list of ints. In case of episodic training, use first item of list as number of shots.")
    parser.add_argument("--n-runs", type=int, default=10000, help="number of few-shot runs")
    parser.add_argument("--n-ways", type=int, default=5, help="number of few-shot ways")
    parser.add_argument("--n-queries", type=int, default=15, help="number of few-shot queries")
    parser.add_argument("--unbalanced-queries", action="store_true", help="Unbalanced queries")
    
    args = parser.parse_args()
        
    if params!=None:
        for key, value in params.items():
            args.__dict__[key]= value
    ### process arguments
    if args.dataset_device == "":
        args.dataset_device = args.device
    if args.dataset_path[-1] != '/':
        args.dataset_path += "/"

    if args.device[:5] == "cuda:" and len(args.device) > 5:
        args.devices = []
        for i in range(len(args.device) - 5):
            args.devices.append(int(args.device[i+5]))
        args.device = args.device[:6]
    else:
        args.devices = [args.device]

    if args.seed == -1:
        args.seed = random.randint(0, 1000000000)

    try:
        n_shots = int(args.n_shots)
        args.n_shots = [n_shots]
    except:
        args.n_shots = eval(args.n_shots)

    if '[' in args.features_path : 
        args.features_path = eval(args.features_path)
    print("args, ", end='')
    return args