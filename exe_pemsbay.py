import argparse
import logging
import torch
import datetime
import json
import yaml
import os
import numpy as np

from dataset_pemsbay import get_dataloader
from main_model import PriSTI_PemsBAY
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
    config["diffusion"]["adj_file"] = 'pems-bay'
    config["seed"] = SEED

    cluster_name = 'pemsbay_vae_latent_mean' # 'pemsbay'
    if cluster_name == 'pemsbay_vae_latent_mean':
        cluster = [
            [
                324, 86, 185, 186, 187, 188, 191, 77, 198, 75, 199, 206, 207, 209, 70, 211, 213, 137,
                220, 222, 63, 234, 58, 56, 184, 89, 180, 178, 130, 129, 139, 141, 124, 123, 122, 118,
                153, 155, 157, 239, 112, 110, 109, 108, 105, 103, 102, 166, 100, 168, 170, 95, 160, 240,
                216, 52, 269, 270, 271, 25, 272, 280, 286, 287, 288, 291, 296, 311, 13, 12, 10, 314,
                7, 316, 5, 317, 322, 265, 264, 136, 51, 34, 248, 246, 45, 36, 50, 255, 257, 46,
                40,
            ],
            [
                190, 312, 156, 228, 154, 315, 150, 148, 143, 233, 245, 142, 237, 140, 319, 321, 230, 309,
                164, 323, 189, 263, 193, 262, 266, 267, 181, 261, 260, 306, 179, 242, 175, 173, 172, 258,
                212, 253, 165, 298, 259, 177, 0, 39, 92, 14, 33, 30, 107, 111, 27, 24, 44, 117,
                62, 119, 78, 47, 65, 125, 76, 18, 22, 16,
            ],
            [
                226, 247, 69, 49, 54, 215, 217, 238, 57, 11, 219, 59, 232, 61, 221, 223, 227, 224,
                236, 313, 229, 252, 301, 300, 299, 297, 294, 15, 290, 289, 20, 284, 283, 282, 250, 281,
                277, 273, 28, 268, 29, 31, 32, 37, 71, 41, 256, 254, 278, 73, 218, 204, 79, 120,
                8, 84, 85, 87, 115, 192, 114, 93, 158, 94, 174, 171, 101, 4, 91, 147, 3, 195,
                131, 202, 128, 127, 194, 144, 200, 99, 196, 197, 146,
            ],
            [
                1, 167, 132, 21, 318, 106, 163, 2, 161, 285, 159, 145, 126, 19, 149, 275, 276, 151,
                116, 279, 23, 35, 113, 274, 97, 98, 38, 81, 80, 225, 64, 74, 305, 201, 214, 67,
                68, 210, 203, 208, 72, 83, 60, 231, 182, 42, 43, 205, 96, 251, 249, 176, 183, 304,
                48, 244, 243, 53, 241, 90, 6,
            ],
            [
                9, 308, 307, 310, 320, 134, 302, 135, 138, 121, 152, 104, 169, 88, 82, 66, 133, 235,
                55, 26, 292, 293, 295, 17, 303, 162,
            ],
        ]
    elif cluster_name == 'pemsbay':
        raise ValueError("The original pemsbay list has 3 clusters; supply a 5-cluster list before enabling it.")
    else:
        raise ValueError(f"Unknown cluster_name: {cluster_name}")
    config['model']['cluster_name'] = cluster_name
    config['model']['cluster'] = cluster


    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
        "./save/pemsbay_" + args.missing_pattern + '_' + current_time + "/"
    )

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # 载入数据
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        config["train"]["batch_size"], device=args.device, missing_pattern=args.missing_pattern,
        is_interpolate=config["model"]["use_guide"], num_workers=args.num_workers, target_strategy=args.targetstrategy
    )
    
    model = PriSTI_PemsBAY(config, args.device, cluster_name).to(args.device)

    model_continue = ""
    #args.modelfolder = 'gote_outsample_20260219_223027'
    if args.modelfolder == "":
        epoch_start = 0
        #model_continue = "gote_outsample_20260220_124602/tmp_model109_notbest.pth"
        if model_continue != "":
            model.load_state_dict(torch.load("./save/" + model_continue, map_location=args.device))
            epoch_start = 110#int(model_continue.split('/')[-1].split('.')[-2][9:])
            
            
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
    parser.add_argument("--missing_pattern", type=str, default="block")     # block|point

    args = parser.parse_args()
    print(args)

    main(args)