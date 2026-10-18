import argparse
import logging
import torch
import datetime
import json
import yaml
import os
import numpy as np

from dataset_metrla import get_dataloader
from main_model import PriSTI_MetrLA
from utils import train, evaluate


def main(args):
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["target_strategy"] = args.targetstrategy
    config["diffusion"]["adj_file"] = 'metr-la'
    config["seed"] = SEED


    cluster_name = 'metrla_vae_latent_mean'  # 'metrla'
    if cluster_name == 'metrla':
        cluster = [
            [
                206, 91, 174, 87, 38, 170, 41, 165, 139, 163, 47, 48, 82, 81, 160, 52, 159, 56,
                141, 155, 60, 76, 63, 150, 73, 148, 133, 95, 144, 191, 12, 127, 96, 15, 120, 193,
                187, 100, 125, 4, 25, 204, 23,
            ],
            [
                75, 113, 111, 142, 74, 124, 140, 80, 119, 115, 112, 94, 104, 205, 130, 138, 137, 101,
                86, 131, 136, 72, 134, 97, 93, 128, 89, 157, 69, 145, 26, 24, 186, 188, 19, 190,
                17, 181, 16, 195, 196, 197, 198, 6, 5, 3, 199, 192, 180, 182, 33, 146, 149, 154,
                61, 116, 57, 158, 54, 162, 161, 43, 42, 166, 172, 175, 34, 164, 117,
            ],
            [
                176, 122, 132, 185, 151, 183, 103, 64, 84, 9, 88, 8, 14, 78, 98, 77, 102, 2,
                28, 59, 40, 70, 29, 68, 83, 79,
            ],
            [
                189, 153, 152, 114, 50, 45, 44, 167, 31, 179, 178, 21, 156, 35, 105, 10, 90, 121,
                203, 99, 18, 106, 126, 200,
            ],
            [
                177, 1, 30, 202, 201, 13, 27, 22, 7, 20, 194, 32, 184, 11, 49, 173, 110, 109,
                123, 108, 107, 129, 92, 135, 85, 143, 71, 147, 67, 66, 65, 62, 58, 55, 53, 51,
                118, 46, 168, 169, 39, 171, 37, 36, 0,
            ],
        ]
    elif cluster_name == 'metrla_vae_latent_mean':
        cluster = [
            [
                67, 79, 78, 74, 161, 162, 68, 136, 66, 65, 64, 167, 168, 169, 171, 173, 176, 45,
                80, 44, 85, 90, 132, 123, 122, 119, 118, 117, 116, 145, 110, 108, 107, 104, 205, 99,
                97, 92, 153, 89, 177, 135, 40, 20, 188, 13, 24, 195, 198, 185, 28, 184, 183, 200,
                201, 203, 31, 178, 182, 34, 2, 32, 1,
            ],
            [
                3, 102, 101, 100, 98, 194, 150, 95, 133, 14, 131, 148, 139, 199, 109, 127, 140, 125,
                124, 151, 112, 146, 5, 137, 18, 154, 164, 63, 41, 61, 60, 165, 57, 53, 170, 181,
                172, 180, 47, 179, 39, 43, 163, 70, 206, 25, 84, 26, 156, 23, 158, 82, 76, 159,
                73, 155, 77,
            ],
            [
                192, 138, 175, 190, 152, 141, 186, 142, 204, 166, 202, 143, 147, 149, 157, 189, 0, 130,
                58, 55, 54, 52, 134, 50, 49, 46, 42, 37, 36, 59, 35, 30, 27, 21, 19, 17,
                11, 10, 9, 8, 7, 6, 33, 62, 103, 71, 129, 128, 126, 121, 120, 115, 114, 113,
                69, 106, 105, 96, 94, 111, 88, 72, 83, 87, 86, 93,
            ],
            [38, 4, 160, 75, 81, 197, 12, 15, 193, 191, 144, 187, 22, 174, 29, 48, 51],
            [196, 56, 91, 16],
        ]
    else:
        raise ValueError(f"Unknown cluster_name: {cluster_name}")
    config['model']['cluster_name'] = cluster_name
    config['model']['cluster'] = cluster


    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
        f"./save/metrla_{cluster_name}_" + args.missing_pattern + '_' + current_time + "/"
    )

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        config["train"]["batch_size"], device=args.device, missing_pattern=args.missing_pattern,
        is_interpolate=config["model"]["use_guide"], num_workers=args.num_workers,
        target_strategy=args.targetstrategy,
    )
    

    model = PriSTI_MetrLA(config, args.device, cluster_name).to(args.device)

    model_continue = ""
    #args.modelfolder = 'metrla_point_20260309_154723'
    if args.modelfolder == "":
        epoch_start = 0
        model_continue = "metrla_point_20260309_154723/tmp_model199.pth"
        if model_continue != "":
            model.load_state_dict(torch.load("./save/" + model_continue, map_location=args.device))
            epoch_start = 180#int(model_continue.split('/')[-1].split('.')[-2][9:])
            
            
        train(
            model,
            config["train"],
            train_loader,
            epoch_start,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PriSTI")
    parser.add_argument("--config", type=str, default="traffic.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument('--num_workers', type=int, default=1, help='Device for Attack')
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument(
        "--targetstrategy", type=str, default="block", choices=["mix", "random", "block"]
    )
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--missing_pattern", type=str, default="point")     # block|point

    args = parser.parse_args()
    print(args)

    main(args)