import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        # 原始大小为int，转为tuple，即：img_size原始输入224，变换后为[224,224]
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)  # 16 x 16
        self.img_size = img_size  # 224 x 224
        self.patch_size = patch_size  # 16 x 16
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 224/16 x 224/16 = 14 x 14
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 14 x 14
        # kernel_size=块大小，即每个块输出一个值，类似每个块展平后使用相同的全连接层进行处理
        # 输入维度为3，输出维度为块向量长度
        # 与原文中：分块、展平、全连接降维保持一致
        # 输出为[B, C, H, W]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)  # 进行 patchty 化
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            "Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # [B, C, H, W] -> [B, C, H*W] ->[B, H*W, C]
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
