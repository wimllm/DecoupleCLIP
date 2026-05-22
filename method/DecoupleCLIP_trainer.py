import cv2
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter

from loss import FocalLoss, BinaryDiceLoss, FocalTverskyLoss
from tools import visualization, calculate_metric, calculate_average_metric
from .DecoupleCLIP_custom_clip import create_model_and_transforms
from .DecoupleCLIP_decoupleclip import DecoupleCLIPModel


class DecoupleCLIP(nn.Module):
    def __init__(
            self,
            backbone,
            feat_list,
            input_dim,
            output_dim,
            learning_rate,
            device,
            image_size,
            adapter_dim=256,
            adapter_dropout=0.0,
            cls_loss_weight=1.0,
            seg_loss_weight=1.0,
            use_last_vit_cls=False,
            last_vit_kernel_size=0,
            last_vit_sigma=0.0,
            last_vit_eps=1e-6,
    ):
        super(DecoupleCLIP, self).__init__()

        self.device = device
        self.feat_list = feat_list
        self.image_size = image_size
        self.cls_loss_weight = cls_loss_weight
        self.seg_loss_weight = seg_loss_weight

        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()
        self.loss_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)

        freeze_clip, _, self.preprocess = create_model_and_transforms(
            backbone,
            image_size,
            pretrained="openai",
        )

        freeze_clip.visual.use_last_vit_cls = use_last_vit_cls
        freeze_clip.visual.last_vit_kernel_size = last_vit_kernel_size
        freeze_clip.visual.last_vit_sigma = last_vit_sigma
        freeze_clip.visual.last_vit_eps = last_vit_eps

        freeze_clip = freeze_clip.to(device)
        freeze_clip.eval()
        for param in freeze_clip.parameters():
            param.requires_grad = False

        self.clip_model = DecoupleCLIPModel(
            freeze_clip=freeze_clip,
            embed_dim=output_dim,
            adapter_dim=adapter_dim,
            adapter_dropout=adapter_dropout,
            output_layers=feat_list,
            device=device,
            image_size=image_size,
        ).to(device)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        self.preprocess.transforms[0] = transforms.Resize(
            size=(image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            max_size=None,
        )
        self.preprocess.transforms[1] = transforms.CenterCrop(size=(image_size, image_size))

        self.learnable_paramter_list = [
            "global_adapter",
            "local_adapter",
            "text_anchor_encoder",
            "local_dwt_decoupling",
        ]

        self.params_to_update = []
        for name, param in self.clip_model.named_parameters():
            for update_name in self.learnable_paramter_list:
                if update_name in name:
                    self.params_to_update.append(param)
                    break

        self.optimizer = torch.optim.AdamW(self.params_to_update, lr=learning_rate, betas=(0.5, 0.999))

    def save(self, path):
        save_dict = {}
        for param, value in self.state_dict().items():
            for update_name in self.learnable_paramter_list:
                if update_name in param:
                    save_dict[param] = value
                    break
        torch.save(save_dict, path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device), strict=False)

    def train_one_batch(self, items):
        image = items["img"].to(self.device)
        cls_name = items["cls_name"]

        anomaly_map, anomaly_score = self.clip_model(image, cls_name, aggregation=False)
        if not isinstance(anomaly_map, list):
            anomaly_map = [anomaly_map]

        gt = items["img_mask"].to(self.device)
        gt = gt.squeeze(1)
        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0

        is_anomaly = items["anomaly"].to(self.device)
        is_anomaly[is_anomaly > 0.5] = 1
        is_anomaly[is_anomaly <= 0.5] = 0

        classification_loss = self.loss_focal(anomaly_score, is_anomaly.unsqueeze(1))

        seg_loss = 0
        for am in anomaly_map:
            seg_loss += (
                self.loss_focal(am, gt)
                + self.loss_tversky(am[:, 1, :, :], gt)
            )

        loss = self.cls_loss_weight * classification_loss + self.seg_loss_weight * seg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train_epoch(self, loader):
        self.clip_model.train()
        loss_list = []
        for items in loader:
            loss = self.train_one_batch(items)
            loss_list.append(loss.item())
        return np.mean(loss_list)

    @torch.no_grad()
    def evaluation(self, dataloader, obj_list, save_fig, save_fig_dir=None):
        self.clip_model.eval()

        results = {
            "cls_names": [],
            "imgs_gts": [],
            "anomaly_scores": [],
            "imgs_masks": [],
            "anomaly_maps": [],
            "imgs": [],
            "names": [],
        }

        with torch.no_grad():
            image_indx = 0
            for indx, items in enumerate(dataloader):
                if save_fig:
                    path = items["img_path"]
                    for _path in path:
                        vis_image = cv2.resize(cv2.imread(_path), (self.image_size, self.image_size))
                        results["imgs"].append(vis_image)
                    cls_name = items["cls_name"]
                    for _cls_name in cls_name:
                        image_indx += 1
                        results["names"].append("{:}-{:03d}".format(_cls_name, image_indx))

                image = items["img"].to(self.device)
                cls_name = items["cls_name"]
                results["cls_names"].extend(cls_name)

                gt_mask = (items["img_mask"] > 0.5).float()
                for _gt_mask in gt_mask:
                    results["imgs_masks"].append(_gt_mask.squeeze(0).numpy())

                anomaly_map, anomaly_score = self.clip_model(image, cls_name, aggregation=True)
                anomaly_map = anomaly_map.cpu().numpy()
                anomaly_score = anomaly_score.cpu().numpy()

                for _anomaly_map, _anomaly_score in zip(anomaly_map, anomaly_score):
                    _anomaly_map = gaussian_filter(_anomaly_map, sigma=4)
                    results["anomaly_maps"].append(_anomaly_map)
                    results["anomaly_scores"].append(_anomaly_score)

                is_anomaly = np.array(items["anomaly"])
                for _is_anomaly in is_anomaly:
                    results["imgs_gts"].append(_is_anomaly)

        if save_fig:
            print("saving fig.....")
            visualization.plot_sample_cv2(
                results["names"],
                results["imgs"],
                {"DecoupleCLIP": results["anomaly_maps"]},
                results["imgs_masks"],
                save_fig_dir,
            )

        metric_dict = dict()
        for obj in obj_list:
            metric_dict[obj] = dict()

        for obj in obj_list:
            metric = calculate_metric(results, obj)
            obj_full_name = f"{obj}"
            metric_dict[obj_full_name] = metric

        metric_dict["Average"] = calculate_average_metric(metric_dict)
        return metric_dict
