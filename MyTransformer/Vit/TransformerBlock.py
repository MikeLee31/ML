import torch.nn as nn

from MyTransformer.Vit.Attention import Attention
from MyTransformer.Vit.DropPath import DropPath
from MyTransformer.Vit.MLP import Mlp


class TransformerBlock(nn.Module):
    '''
    Transformer Block combines both the attention module and the feed forward module with layer
    normalization, dropout and residual connections. The sequence of operations is as follows :-

    Input -> LayerNorm1 -> Attention -> Residual -> LayerNorm2 -> FeedForward -> Output
      |                                   |  |                                      |
      |-------------Addition--------------|  |---------------Addition---------------|
    '''

    def __init__(self,
                 embed_dim,
                 heads=8,
                 mlp_ratio=4,  # mlp为4倍
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 activation=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(embed_dim, heads=heads,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 这里可以选择 drop path， 或者Drop out， 在官方代码中使用了Drop path
        self.drop_path = DropPath(drop_path_ratio)
        # self.drop = nn.Dropout(drop_path_ratio)
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=activation, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
