import argparse
import torch
import datetime
import json
import yaml
import os
import logging
import numpy as np

from dataset_gote import get_dataloader
from main_model import PriSTI_gote
from utils import train, evaluate


# def main(args):


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PriSTI")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument('--num_workers', type=int, default=1, help='Device for Attack')
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument(
        "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
    )
    parser.add_argument(
        "--val_len", type=float, default=0.1, help="the ratio of data used for validation (value:[0-1])"
    )
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--cluster_name", default='gote')

    args = parser.parse_args()
    print(args)



    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["target_strategy"] = args.targetstrategy
    config["diffusion"]["adj_file"] = 'gote'
    config["seed"] = SEED


    cluster_name = args.cluster_name
    #cluster_name = 'gote' #  Corr
    #cluster_name = 'gote_vae_latent_mean'  # VAE
    #cluster_name = 'gote_random_control'
    #cluster_name = 'gote_k_1'
    #cluster_name = 'gote_k_2'
    #cluster_name = 'gote_k_4'
    #cluster_name = 'gote_k_5'
    if cluster_name == 'gote':
        cluster = [
            [
                0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 22,
                23, 24, 25, 26, 28, 29, 31, 32, 34, 35, 36, 37, 39, 43, 44, 47, 49, 50,
                52,
            ],
            [15, 19, 21, 38],
            [1, 7, 27, 30, 33, 40, 41, 42, 45, 46, 48, 51],
        ]
    elif cluster_name == 'gote_vae_latent_mean':
        cluster = [
            [0, 4, 5, 6, 7, 10, 11, 12, 14, 18, 23, 25, 26, 28, 30, 32, 44, 52],
            [9, 13, 22, 31],
            [1, 2, 3, 8, 15, 16, 17, 19, 20, 21, 24, 27, 29, 33, 34, 35, 36, 37,
                38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51],
        ]
    elif cluster_name == 'gote_k_2':
        cluster = [[ 0,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 16, 17, 18, 20,
                     22, 23, 24, 25, 26, 28, 29, 31, 32, 34, 35, 36, 37, 39, 43, 44, 47,
                     49, 50, 52],
                   [ 1,  7, 15, 19, 21, 27, 30, 33, 38, 40, 41, 42, 45, 46, 48, 51]]
    elif cluster_name == 'gote_k_4':
        cluster = [[ 0,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 16, 17, 18, 20,
                     22, 23, 24, 25, 26, 28, 29, 31, 32, 34, 35, 36, 37, 39, 43, 44, 47,
                     49, 50, 52],
                   [1, 30, 33, 41, 45, 46, 51],
                   [7, 27, 40, 42, 48],
                   [15, 19, 21, 38]]
    elif cluster_name == 'gote_k_5':
        cluster = [[ 0,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 16, 17, 18, 20,
		       22, 23, 24, 25, 26, 28, 29, 31, 32, 34, 35, 36, 37, 39, 43, 44, 47,
		       49, 50, 52],
                 [1, 30, 33, 41, 45, 46, 51],
		 [7, 27, 40, 42, 48],
		 [15],
		 [19, 21, 38]]

    elif cluster_name == 'gote_random_control':
        a = np.arange(53)
        np.random.shuffle(a)
        cluster = [a[:37].tolist(),
        	a[37:41].tolist(),
        	a[41:].tolist()]
    elif cluster_name == 'gote_k_1':
        a = np.arange(53)
        cluster =[a.tolist()]
    else:
        raise ValueError(f"Unknown cluster_name: {cluster_name}")
    config['model']['cluster_name'] = cluster_name
    config['model']['cluster'] = cluster
    
    
    
    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
        f"./save/gote_outsample_{cluster_name}_seed{SEED}_" + current_time + "/"
    )

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        config["train"]["batch_size"], device=args.device, val_len=args.val_len,
        is_interpolate=config["model"]["use_guide"], num_workers=args.num_workers,
        target_strategy=args.targetstrategy, mask_sensor=config["model"]["mask_sensor"]
    )
    

    model = PriSTI_gote(config, args.device, cluster_name, target_dim=53).to(args.device)
            
    model_continue = ""
    #args.modelfolder = 'gote_outsample_20260219_223027'
    if args.modelfolder == "":
        
        epoch_start = 0
        #model_continue = "gote_outsample_gote_k_5_seed42_20261016_084908/tmp_model189_notbest.pth"
        model_continue = ""
        if model_continue != "":
            model.load_state_dict(torch.load("./save/" + model_continue, map_location=args.device))
            epoch_start = 190#int(model_continue.split('/')[-1].split('.')[-2][9:])
            
        train(
            model,
            config["train"],
            train_loader,
            epoch_start = epoch_start,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth", map_location=args.device))

    logging.basicConfig(filename=foldername + '/test_model.log', level=logging.DEBUG)
    logging.info("model_name={}".format(args.modelfolder))
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )
    
    
    # main(args)