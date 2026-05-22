import argparse
import json
import os
import warnings

import torch
from tqdm import tqdm

from dataset import get_data
from method.DecoupleCLIP_trainer import DecoupleCLIP
from tools import Logger, log_metrics, write2csv
from tools.DecoupleCLIP_training_tools import setup_paths, setup_seed

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

setup_seed(111)


def train(args):
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_fig = args.save_fig

    model_name, image_dir, csv_path, log_path, ckp_path, tensorboard_logger = setup_paths(args)
    logger = Logger(log_path)

    for key, value in sorted(vars(args).items()):
        logger.info(f"{key} = {value}")
    logger.info("Model name: {:}".format(model_name))

    config_path = os.path.join("./model_configs", f"{args.model}.json")
    with open(config_path, "r") as f:
        model_configs = json.load(f)

    n_layers = model_configs["vision_cfg"]["layers"]
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]

    model = DecoupleCLIP(
        backbone=args.model,
        feat_list=features_list,
        input_dim=model_configs["vision_cfg"]["width"],
        output_dim=model_configs["embed_dim"],
        learning_rate=learning_rate,
        device=device,
        image_size=image_size,
        adapter_dim=args.adapter_dim,
        adapter_dropout=args.adapter_dropout,
        cls_loss_weight=args.cls_loss_weight,
        seg_loss_weight=args.seg_loss_weight,
        use_last_vit_cls=args.use_last_vit_cls,
        last_vit_kernel_size=args.last_vit_kernel_size,
        last_vit_sigma=args.last_vit_sigma,
        last_vit_eps=args.last_vit_eps,
    ).to(device)

    train_data_cls_names, train_data, train_data_root = get_data(
        dataset_type_list=args.training_data,
        transform=model.preprocess,
        target_transform=model.transform,
        training=True,
    )

    test_data_cls_names, test_data, test_data_root = get_data(
        dataset_type_list=args.testing_data,
        transform=model.preprocess,
        target_transform=model.transform,
        training=False,
    )

    logger.info("Data Root: training, {:}; testing, {:}".format(train_data_root, test_data_root))

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    best_auroc_px = -1e1
    for epoch in tqdm(range(epochs)):
        loss = model.train_epoch(train_dataloader)

        if (epoch + 1) % args.print_freq == 0:
            logger.info("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, epochs, loss))
            tensorboard_logger.add_scalar("loss", loss, epoch)

        if (epoch + 1) % args.valid_freq == 0 or (epoch == epochs - 1):
            save_fig_flag = save_fig if epoch == epochs - 1 else False

            logger.info("=============================Testing ====================================")
            metric_dict = model.evaluation(
                test_dataloader,
                test_data_cls_names,
                save_fig_flag,
                image_dir,
            )

            log_metrics(metric_dict, logger, tensorboard_logger, epoch)

            auroc_px = metric_dict["Average"]["auroc_px"]
            if auroc_px > best_auroc_px:
                for k in metric_dict.keys():
                    write2csv(metric_dict[k], test_data_cls_names, k, csv_path)

                ckp_path_best = ckp_path + "_best.pth"
                model.save(ckp_path_best)
                best_auroc_px = auroc_px
                logger.info("Saved best checkpoint by Average/P-AUROC: {:.2f}".format(best_auroc_px))


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DecoupleCLIP", add_help=True)

    parser.add_argument("--training_data", type=str, default=["mvtec"], nargs="+",
                        help="Datasets for training (default: ['mvtec'])")
    parser.add_argument("--testing_data", type=str, default="visa", help="Dataset for testing (default: 'visa')")
    parser.add_argument("--save_path", type=str, default="./workspaces",
                        help="Directory to save results (default: './workspaces')")
    parser.add_argument("--model", type=str, default="ViT-L-14-336",
                        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336","ViT-bigG-14"],
                        help="The CLIP model to be used (default: 'ViT-L-14-336')")
    parser.add_argument("--save_fig", type=str2bool, default=False,
                        help="Save figures for visualizations (default: False)")

    parser.add_argument("--exp_indx", type=int, default=39, help="Index of the experiment (default: 0)")
    parser.add_argument("--epoch", type=int, default=15, help="Number of epochs (default: 5)")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--image_size", type=int, default=518, help="Size of the input images (default: 518)")
    parser.add_argument("--print_freq", type=int, default=1, help="Frequency of print statements (default: 1)")
    parser.add_argument("--valid_freq", type=int, default=1, help="Frequency of validation (default: 1)")

    parser.add_argument("--adapter_dim", type=int, default=768, help="Adapter bottleneck dimension (default: 768)")
    parser.add_argument("--adapter_dropout", type=float, default=0.0, help="Adapter dropout (default: 0.0)")
    parser.add_argument("--cls_loss_weight", type=float, default=1.0,
                        help="Image classification loss weight (default: 1.0)")
    parser.add_argument("--seg_loss_weight", type=float, default=1.0,
                        help="Pixel segmentation loss weight (default: 1.0)")
    parser.add_argument("--use_last_vit_cls", type=str2bool, default=False,
                        help="Use LAST-ViT frequency-domain patch selection for the global CLS feature")
    parser.add_argument("--last_vit_kernel_size", type=int, default=0,
                        help="Gaussian kernel size for LAST-ViT CLS selection. 0 uses visual hidden dim")
    parser.add_argument("--last_vit_sigma", type=float, default=0.0,
                        help="Gaussian sigma for LAST-ViT CLS selection. 0 uses sqrt(visual hidden dim)")
    parser.add_argument("--last_vit_eps", type=float, default=1e-6,
                        help="Numerical epsilon for LAST-ViT CLS selection (default: 1e-6)")

    train(parser.parse_args())
