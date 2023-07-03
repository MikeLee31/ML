from functools import partial

import torch
import torch.nn as nn

from .PatchEmbed import PatchEmbed
from .TransformerBlock import TransformerBlock


class VisionTransformer(nn.Module):
    '''
    Vision Transformer is the complete end to end model architecture which combines all the above modules
    in a sequential manner. The sequence of the operations is as follows -

    Input -> CreatePatches -> ClassToken, PatchToEmbed , PositionEmbed -> Transformer -> ClassificationHead -> Output
                                   |            | |                |
                                   |---Concat---| |----Addition----|
    '''

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_c=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 heads=12,
                 mlp_ratio=4.0,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None):
        super().__init__()
        self.name = 'VisionTransformer'
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 位置编码 (1,embedim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # 加上类别
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 等差数列的drop_path_ratio
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim=embed_dim, heads=heads, mlp_ratio=mlp_ratio,
                             drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                             norm_layer=norm_layer, activation=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        # x (batch_size, seq_len+1, embed_dim)

        x = self.norm(x)
        return self.pre_logits(x[:, 0])  # batch_size, embed_dim)

    def forward(self, x):
        x = self.forward_features(x)
        # (batch_size, embed_dim)
        x = self.head(x)
        # (batch_size, classes)

        return x

    def _init_vit_weights(m):
        """
        ViT weight initialization
        :param m: module
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              heads=12,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              heads=12,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              heads=16,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224(num_classes: int = 21843):
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              heads=16,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224(num_classes: int = 21843):
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              heads=16,
                              num_classes=num_classes)
    return model


def vit_cifar_patch4_32(num_classes: int = 10):
    model = VisionTransformer(img_size=32,
                              patch_size=2,
                              embed_dim=512,
                              depth=12,
                              heads=16,
                              num_classes=num_classes)
    return model
