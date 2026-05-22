import argparse
import json
import os
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from dataset import dataset_dict, get_data
from method.DecoupleCLIP_trainer import DecoupleCLIP
from tools import Logger, setup_seed, write2csv

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

setup_seed(111)


def test(args):
    assert os.path.isfile(args.ckt_path), f"Please check the path of pre-trained model, {args.ckt_path} is not valid."

    batch_size = args.batch_size
    image_size = args.image_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_fig = args.save_fig
    logger = Logger("DecoupleCLIP_log.txt")

    for key, value in sorted(vars(args).items()):
        logger.info(f"{key} = {value}")

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
        learning_rate=0.0,
        device=device,
        image_size=image_size,
        adapter_dim=args.adapter_dim,
        adapter_dropout=args.adapter_dropout,
        cls_loss_weight=1.0,
        seg_loss_weight=1.0,
        use_last_vit_cls=args.use_last_vit_cls,
        last_vit_kernel_size=args.last_vit_kernel_size,
        last_vit_sigma=args.last_vit_sigma,
        last_vit_eps=args.last_vit_eps,
    ).to(device)
    model.load(args.ckt_path)

    if args.testing_model == "dataset":
        assert args.testing_data in dataset_dict.keys(), (
            f"You entered {args.testing_data}, but we only support {dataset_dict.keys()}"
        )

        save_root = args.save_path
        csv_root = os.path.join(save_root, "csvs")
        image_root = os.path.join(save_root, "images")
        os.makedirs(csv_root, exist_ok=True)
        os.makedirs(image_root, exist_ok=True)
        csv_path = os.path.join(csv_root, f"DecoupleCLIP-{args.testing_data}.csv")
        image_dir = os.path.join(image_root, f"DecoupleCLIP-{args.testing_data}")
        os.makedirs(image_dir, exist_ok=True)

        test_data_cls_names, test_data, _ = get_data(
            dataset_type_list=args.testing_data,
            transform=model.preprocess,
            target_transform=model.transform,
            training=False,
        )

        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        metric_dict = model.evaluation(test_dataloader, test_data_cls_names, save_fig, image_dir)

        for tag, data in metric_dict.items():
            logger.info(
                "{:>15} \t\tI-Auroc:{:.2f} \tI-F1:{:.2f} \tI-AP:{:.2f} "
                "\tP-Auroc:{:.2f} \tP-F1:{:.2f} \tP-AP:{:.2f}".format(
                    tag,
                    data["auroc_im"],
                    data["f1_im"],
                    data["ap_im"],
                    data["auroc_px"],
                    data["f1_px"],
                    data["ap_px"],
                )
            )

        for k in metric_dict.keys():
            write2csv(metric_dict[k], test_data_cls_names, k, csv_path)

    elif args.testing_model == "image":
        assert os.path.isfile(args.image_path), f"Please verify the input image path: {args.image_path}"
        os.makedirs(args.save_path, exist_ok=True)

        ori_image = cv2.resize(cv2.imread(args.image_path), (args.image_size, args.image_size))
        pil_img = Image.open(args.image_path).convert("RGB")

        img_input = model.preprocess(pil_img).unsqueeze(0).to(model.device)
        with torch.no_grad():
            anomaly_map, anomaly_score = model.clip_model(img_input, [args.class_name], aggregation=True)

        anomaly_map = anomaly_map[0, :, :].cpu().numpy()
        anomaly_score = anomaly_score[0].cpu().numpy()
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        anomaly_map = (anomaly_map * 255).astype(np.uint8)

        heat_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        vis_map = cv2.addWeighted(heat_map, 0.5, ori_image, 0.5, 0)
        vis_map = cv2.hconcat([ori_image, vis_map])

        save_path = os.path.join(args.save_path, args.save_name)
        print(f"Anomaly detection results are saved in {save_path}, with an anomaly of {anomaly_score:.3f} ")
        cv2.imwrite(save_path, vis_map)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DecoupleCLIP", add_help=True)

    parser.add_argument("--ckt_path", type=str, default="weights/best.pth",
                        help="Path to the pre-trained model")
    parser.add_argument("--testing_model", type=str, default="dataset", choices=["dataset", "image"],
                        help="Model for testing (default: 'dataset')")
    parser.add_argument("--testing_data", type=str, default="visa", help="Dataset for testing (default: 'visa')")
    parser.add_argument("--image_path", type=str, default="asset/img.png",
                        help="Path of the testing image (default: 'asset/img.png')")
    parser.add_argument("--class_name", type=str, default="candle",
                        help="The class name of the testing image (default: 'candle')")
    parser.add_argument("--save_name", type=str, default="test.png",
                        help="Saved visualization name (default: 'test.png')")
    parser.add_argument("--save_path", type=str, default="./workspaces",
                        help="Directory to save results (default: './workspaces')")
    parser.add_argument("--model", type=str, default="ViT-L-14-336",
                        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"],
                        help="The CLIP model to be used (default: 'ViT-L-14-336')")
    parser.add_argument("--save_fig", type=str2bool, default=False,
                        help="Save figures for visualizations (default: False)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--image_size", type=int, default=518, help="Size of the input images (default: 518)")
    parser.add_argument("--adapter_dim", type=int, default=768, help="Adapter bottleneck dimension (default: 768)")
    parser.add_argument("--adapter_dropout", type=float, default=0.0, help="Adapter dropout (default: 0.0)")
    parser.add_argument("--use_last_vit_cls", type=str2bool, default=False,
                        help="Use LAST-ViT frequency-domain patch selection for the global CLS feature")
    parser.add_argument("--last_vit_kernel_size", type=int, default=0,
                        help="Gaussian kernel size for LAST-ViT CLS selection. 0 uses visual hidden dim")
    parser.add_argument("--last_vit_sigma", type=float, default=0.0,
                        help="Gaussian sigma for LAST-ViT CLS selection. 0 uses sqrt(visual hidden dim)")
    parser.add_argument("--last_vit_eps", type=float, default=1e-6,
                        help="Numerical epsilon for LAST-ViT CLS selection (default: 1e-6)")

    test(parser.parse_args())
