import math
from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F

from .simple_tokenizer import SimpleTokenizer


class ResMLP(nn.Module):
    def __init__(self, c_in: int, reduction: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden_dim = max(c_in // reduction, 1)
        self.down = nn.Linear(c_in, hidden_dim, bias=False)
        self.bn = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(hidden_dim, c_in, bias=False)

    def _fc(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.up(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in (2, 3):
            raise ValueError(f"ResMLP expects a 2D or 3D tensor, but got shape {tuple(x.shape)}.")
        return x + self._fc(x)


class GlobalVisualAdapter(nn.Module):
    def __init__(self, input_dim: int, reduction: int = 2, dropout: float = 0.0):
        super().__init__()
        self.adapter = ResMLP(input_dim, reduction=reduction, dropout=dropout)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.adapter(image_features)


class LocalVisualAdapter(nn.Module):
    def __init__(self, input_dim: int, reduction: int = 2, dropout: float = 0.0):
        super().__init__()
        self.adapter = ResMLP(input_dim, reduction=reduction, dropout=dropout)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        return self.adapter(patch_features)


class FFTHighFrequencyDecoupling(nn.Module):
    def __init__(
            self,
            dim: int,
            grid_size: int,
            num_groups: int = 8,
            hp_init_radius: float = 0.15,
            use_residual: bool = True,
            eps: float = 1e-6,
    ):
        super().__init__()
        if dim % num_groups != 0:
            num_groups = 1
        self.dim = dim
        self.grid_size = grid_size
        self.num_groups = num_groups
        self.use_residual = use_residual
        self.eps = eps

        h = grid_size
        w = grid_size // 2 + 1
        self.weight = nn.Parameter(torch.randn(num_groups, h, w, 2) * 0.02)

        gy = torch.linspace(-1.0, 1.0, h)
        gx = torch.linspace(0.0, 1.0, w)
        gy, gx = torch.meshgrid(gy, gx, indexing="ij")
        hp_mask = ((gy ** 2 + gx ** 2).sqrt() > hp_init_radius).float()
        self.register_buffer("hp_mask", hp_mask)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        if patch_features.dim() != 3:
            raise ValueError(
                f"FFTHighFrequencyDecoupling expects a 3D tensor, but got shape {tuple(patch_features.shape)}."
            )
        b, l, d = patch_features.shape
        h = int(math.sqrt(l))
        if h * h != l:
            raise ValueError(f"Patch token length {l} is not a square number.")
        if h != self.grid_size:
            raise ValueError(
                f"FFTHighFrequencyDecoupling expects grid {self.grid_size}, but got {h}."
            )
        if d != self.dim:
            raise ValueError(f"Expected feature dim {self.dim}, got {d}.")

        residual = patch_features
        orig_dtype = patch_features.dtype
        x = patch_features.view(b, h, h, d).permute(0, 3, 1, 2).contiguous()
        x = x.float()
        Xf = torch.fft.rfft2(x, norm="ortho")

        dg = d // self.num_groups
        Xf = Xf.view(b, self.num_groups, dg, h, h // 2 + 1)
        w = torch.view_as_complex(self.weight)
        gain = w.unsqueeze(0).unsqueeze(2) + self.hp_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        Xf = Xf * gain
        Xf = Xf.reshape(b, d, h, h // 2 + 1)

        x_out = torch.fft.irfft2(Xf, s=(h, h), norm="ortho")
        x_out = x_out.to(orig_dtype)
        out = x_out.permute(0, 2, 3, 1).contiguous().view(b, l, d)

        if self.use_residual:
            return residual + self.alpha * out
        return self.alpha * out


class TextAnchorAdapter(nn.Module):
    def __init__(self, embed_dim: int, adapter_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, adapter_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(adapter_dim, embed_dim)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, anchors: torch.Tensor) -> torch.Tensor:
        residual = anchors
        x = self.norm(anchors)
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + x


class TextAnchorEncoder(nn.Module):
    def __init__(self, embed_dim: int, adapter_dim: int = 256, adapter_dropout: float = 0.0):
        super().__init__()
        self.tokenizer = SimpleTokenizer()
        self.global_cache = {}
        self.local_cache = {}
        self.embed_dim = embed_dim
        self.global_text_adapter = TextAnchorAdapter(
            embed_dim=embed_dim,
            adapter_dim=adapter_dim,
            dropout=adapter_dropout,
        )
        self.local_text_adapter = TextAnchorAdapter(
            embed_dim=embed_dim,
            adapter_dim=adapter_dim,
            dropout=adapter_dropout,
        )

        self.template_list = [
            "a cropped photo of the {}.",
            "a close-up photo of a {}.",
            "a close-up photo of the {}.",
            "a bright photo of a {}.",
            "a bright photo of the {}.",
            "a dark photo of the {}.",
            "a dark photo of a {}.",
            "a jpeg corrupted photo of the {}.",
            "a jpeg corrupted photo of the {}.",
            "a blurry photo of the {}.",
            "a blurry photo of a {}.",
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a small {}.",
            "a photo of the small {}.",
            "a photo of a large {}.",
            "a photo of the large {}.",
            "a photo of the {} for visual inspection.",
            "a photo of a {} for visual inspection.",
            "a photo of the {} for anomaly detection.",
            "a photo of a {} for anomaly detection.",
        ]
        self.static_global_normal_list = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage",
        ]
        self.static_global_anomaly_list = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage",
        ]
        self.static_local_normal_list = [
            "{}",
            "{} with normal texture",
            "{} with regular texture",
            "{} with uniform texture",
            "{} with consistent texture",
            "{} with clean surface",
            "{} with intact surface",
            "{} with smooth surface",
            "{} with regular pattern",
            "{} with consistent color",
            "{} without local defect",
            "{} without damaged region",
            "{} without abnormal region",
        ]
        self.static_local_anomaly_list = [
            "{} with abnormal region",
            "{} with anomalous region",
            "{} with damaged region",
            "{} with defective region",
            "{} with local defect",
            "{} with surface flaw",
            "{} with abnormal texture",
            "{} with irregular texture",
            "{} with rough texture",
            "{} with scratch",
            "{} with crack",
            "{} with stain",
            "{} with contamination",
            "{} with broken area",
            "{} with missing part",
            "{} with color anomaly",
            "{} with structural anomaly",
        ]

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]

        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return result

    def _encode_prompts(self, model: nn.Module, prompts: List[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenize(prompts, context_length=77).to(device)
        features = model.encode_text(tokens)
        features = F.normalize(features, dim=-1, eps=1e-6)
        return features.detach()

    def _build_prompts(self, cls_name: str, state_list: List[str]) -> List[str]:
        prompts = []
        for state in state_list:
            state_description = state.format(cls_name)
            for template in self.template_list:
                prompts.append(template.format(state_description))
        return prompts

    def _encode_anchor_pair(
            self,
            model: nn.Module,
            cls_name: str,
            normal_list: List[str],
            anomaly_list: List[str],
            device: torch.device,
    ) -> torch.Tensor:
        normal_prompts = self._build_prompts(cls_name, normal_list)
        anomaly_prompts = self._build_prompts(cls_name, anomaly_list)

        normal_features = self._encode_prompts(model, normal_prompts, device).mean(dim=0)
        anomaly_features = self._encode_prompts(model, anomaly_prompts, device).mean(dim=0)

        normal_features = F.normalize(normal_features, dim=-1, eps=1e-6)
        anomaly_features = F.normalize(anomaly_features, dim=-1, eps=1e-6)
        return torch.stack([normal_features, anomaly_features], dim=0)

    def forward(self, model: nn.Module, cls_names: List[str], device: torch.device):
        global_features = []
        local_features = []

        for cls_name in cls_names:
            cls_name = cls_name.replace("-", " ")
            if cls_name not in self.global_cache:
                with torch.no_grad():
                    self.global_cache[cls_name] = self._encode_anchor_pair(
                        model,
                        cls_name,
                        self.static_global_normal_list,
                        self.static_global_anomaly_list,
                        device,
                    )
            if cls_name not in self.local_cache:
                with torch.no_grad():
                    self.local_cache[cls_name] = self._encode_anchor_pair(
                        model,
                        cls_name,
                        self.static_local_normal_list,
                        self.static_local_anomaly_list,
                        device,
                    )

            static_global = self.global_cache[cls_name].to(device)
            static_local = self.local_cache[cls_name].to(device)

            global_anchor = self.global_text_adapter(static_global)
            local_anchor = self.local_text_adapter(static_local)

            global_anchor = F.normalize(global_anchor, dim=-1, eps=1e-6)
            local_anchor = F.normalize(local_anchor, dim=-1, eps=1e-6)

            global_features.append(global_anchor)
            local_features.append(local_anchor)

        global_features = torch.stack(global_features, dim=0)
        local_features = torch.stack(local_features, dim=0)
        return global_features, local_features


class DecoupleCLIPModel(nn.Module):
    def __init__(
            self,
            freeze_clip: nn.Module,
            embed_dim: int,
            adapter_dim: int,
            adapter_dropout: float,
            output_layers: list,
            device: str,
            image_size: int,
    ):
        super().__init__()
        self.freeze_clip = freeze_clip
        self.visual = freeze_clip.visual
        self.transformer = freeze_clip.transformer
        self.token_embedding = freeze_clip.token_embedding
        self.positional_embedding = freeze_clip.positional_embedding
        self.ln_final = freeze_clip.ln_final
        self.text_projection = freeze_clip.text_projection
        self.attn_mask = freeze_clip.attn_mask

        self.embed_dim = embed_dim
        self.output_layers = output_layers
        self.device = device
        self.image_size = image_size

        self.text_anchor_encoder = TextAnchorEncoder(
            embed_dim=embed_dim,
            adapter_dim=adapter_dim,
            adapter_dropout=adapter_dropout,
        )
        if adapter_dim <= 0:
            raise ValueError(f"adapter_dim must be positive, but got {adapter_dim}.")
        adapter_reduction = max(embed_dim // adapter_dim, 1)
        self.global_adapter = GlobalVisualAdapter(embed_dim, reduction=adapter_reduction, dropout=adapter_dropout)
        patch_size = self.visual.conv1.kernel_size[0]
        grid_size = image_size // patch_size
        num_layers = len(output_layers)
        self.local_dwt_decoupling = nn.ModuleList([
            FFTHighFrequencyDecoupling(
                dim=embed_dim,
                grid_size=grid_size,
                use_residual=True,
            )
            for _ in range(num_layers)
        ])
        self.local_adapter = nn.ModuleList([
            LocalVisualAdapter(embed_dim, reduction=adapter_reduction, dropout=adapter_dropout)
            for _ in range(num_layers)
        ])

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)

        for r in self.transformer.resblocks:
            x, _ = r(x, attn_mask=self.attn_mask)

        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        eot_indices = text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot_indices] @ self.text_projection
        return x

    def encode_image(self, image: torch.Tensor):
        with torch.no_grad():
            query_feats, query_patch_feats = self.visual.forward(image, self.output_layers)
        return query_feats.detach(), [patch_feat.detach() for patch_feat in query_patch_feats]

    @staticmethod
    def _local_similarity(local_features: torch.Tensor, local_text_features: torch.Tensor) -> torch.Tensor:
        b, l, _ = local_features.shape
        h = int(math.sqrt(l))
        if h * h != l:
            raise ValueError(f"Patch token length {l} is not a square number.")

        local_features = F.normalize(local_features, dim=-1, eps=1e-6)
        local_text_features = F.normalize(local_text_features, dim=-1, eps=1e-6)
        logits = 100.0 * torch.einsum("ble,bce->blc", local_features, local_text_features)
        logits = torch.clamp(logits, min=-80.0, max=80.0)
        logits = logits.permute(0, 2, 1).contiguous().view(b, 2, h, h)
        return logits

    def visual_text_similarity(
            self,
            query_feats: torch.Tensor,
            query_patch_feats: List[torch.Tensor],
            global_text_features: torch.Tensor,
            local_text_features: torch.Tensor,
            aggregation: bool,
    ):
        global_features = self.global_adapter(query_feats)
        global_features = F.normalize(global_features, dim=-1, eps=1e-6)
        global_text_features = F.normalize(global_text_features, dim=-1, eps=1e-6)
        anomaly_score = 100.0 * torch.einsum("be,bce->bc", global_features, global_text_features)
        anomaly_score = torch.clamp(anomaly_score, min=-80.0, max=80.0)
        anomaly_score = torch.softmax(anomaly_score, dim=1)

        if not query_patch_feats:
            raise ValueError("No patch features were returned by the visual backbone.")
        if len(query_patch_feats) != len(self.local_adapter):
            raise ValueError(
                f"Got {len(query_patch_feats)} layers from backbone but model expects "
                f"{len(self.local_adapter)} layers."
            )

        layer_logits = []
        for i, patch_feat in enumerate(query_patch_feats):
            feat = self.local_dwt_decoupling[i](patch_feat)
            feat = self.local_adapter[i](feat)
            feat = F.normalize(feat, dim=-1, eps=1e-6)
            logits = self._local_similarity(feat, local_text_features)
            logits = F.interpolate(
                logits,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=True,
            )
            layer_logits.append(logits)

        if aggregation:
            prob_stack = torch.stack([torch.softmax(l, dim=1) for l in layer_logits], dim=0)
            prob = prob_stack.mean(dim=0)
            anomaly_map = (prob[:, 1:, :, :] + 1 - prob[:, 0:1, :, :]) / 2.0
            return anomaly_map.squeeze(1), anomaly_score[:, 1]

        anomaly_maps = [torch.softmax(l, dim=1) for l in layer_logits]
        return anomaly_maps, anomaly_score

    def extract_feat(self, image: torch.Tensor, cls_name: List[str]):
        query_feats, query_patch_feats = self.encode_image(image)
        global_text_features, local_text_features = self.text_anchor_encoder(self, cls_name, image.device)
        return query_feats, query_patch_feats, global_text_features, local_text_features

    def forward(self, image: torch.Tensor, cls_name: List[str], aggregation: bool = True):
        query_feats, query_patch_feats, global_text_features, local_text_features = self.extract_feat(image, cls_name)
        return self.visual_text_similarity(
            query_feats,
            query_patch_feats,
            global_text_features,
            local_text_features,
            aggregation,
        )
