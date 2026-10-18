import argparse
import torch
import datetime
import json
import yaml
import os
import logging
import numpy as np

from dataset_aqi36 import get_dataloader
from main_model import PriSTI_aqi36
from utils import train, evaluate


# def main(args):


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PriSTI")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument('--num_workers', type=int, default=16, help='Device for Attack')
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument(
        "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
    )
    parser.add_argument(
        "--val_len", type=float, default=0.1, help="the ratio of data used for validation (value:[0-1])"
    )
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42) #38,39,40,41,42
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--cluster_name", default='pm25')

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
    config["diffusion"]["adj_file"] = 'AQI36'
    config["seed"] = SEED



    cluster_name = args.cluster_name
    #cluster_name = 'pm25' # by correlation
    #cluster_name = 'pm25_vae_latent_mean'   # by VAE
    #cluster_name = 'pm25_random_control'  # for ablation
    #cluster_name = 'pm25_k_1'
    #cluster_name = 'pm25_k_2'
    #cluster_name = 'pm25_k_4'
    #cluster_name = 'pm25_k_5'
    if cluster_name == 'pm25':
        cluster = [
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [0, 1, 22, 23, 24, 26, 27, 28, 29, 31],
            [25, 30, 32, 33, 34, 35],        
        ]
    elif cluster_name == 'pm25_vae_latent_mean':
        cluster = [
            [0, 22, 21, 20, 19, 15, 13, 12, 11, 14, 8, 7, 6, 5, 9, 3, 2],
            [32, 26, 25, 24, 23, 28, 29, 30, 31, 27, 1],
            [33, 17, 34, 16, 10, 4, 18, 35],
        ]
    elif cluster_name == 'pm25_k_1':
        cluster = [
            np.arange(36).tolist(),
        ]
    elif cluster_name == 'pm25_k_2':
        cluster = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 31, 32],
            [28,30,33,34,35],
        ]
    elif cluster_name == 'pm25_k_4':
        cluster = [
	    [0, 1, 22, 23, 24, 25, 26, 27, 29, 31, 32],
	    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
	    [28, 30],
	    [33, 34, 35],
	]
    elif cluster_name == 'pm25_k_5':
        cluster = [
	    [0, 1, 22, 23, 24, 26, 27, 29, 31],
	    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
	    [25, 32],
	    [28, 30],
	    [33, 34, 35],
	]
    elif cluster_name == 'pm25_random_control':
        a = np.arange(36)
        np.random.shuffle(a)
        cluster = [a[:20].tolist(),
        	a[20:30].tolist(),
        	a[30:].tolist()]
    else:
        raise ValueError(f"Unknown cluster_name: {cluster_name}")
    config['model']['cluster_name'] = cluster_name
    config['model']['cluster'] = cluster
    
    
    print(json.dumps(config, indent=4))
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
        f"./save/pm25_outsample_{cluster_name}_seed{SEED}_" + current_time + "/"
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
    
    
    
    model = PriSTI_aqi36(config, args.device, cluster_name).to(args.device)
    
    #args.modelfolder = 'pm25_outsample_pm25_k_2_seed42_20261007_194740'
    if args.modelfolder == "":
        
        epoch_start = 0
        #model_continue = "pm25_outsample_pm25_seed38_20261002_191825/tmp_model199_notbest.pth"
        model_continue = ""
        if model_continue != "":
            model.load_state_dict(torch.load("./save/" + model_continue, map_location=args.device))
            epoch_start = 199#int(model_continue.split('/')[-1].split('.')[-2][9:])
            
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