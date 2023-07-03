import torch.nn as nn


class Attention(nn.Module):
    '''
    Attention Module used to perform self-attention operation allowing the model to attend
    information from different representation subspaces on an input sequence of embeddings.
    The sequence of operations is as follows :-

    Input -> Query, Key, Value -> ReshapeHeads -> Query.TransposedKey -> Softmax -> Dropout
    -> AttentionScores.Value -> ReshapeHeadsBack -> Output
    '''

    def __init__(self,
                 embed_dim,  # 输入token的dim
                 heads=8,
                 activation=None,
                 attn_drop_ratio=0.,  # dropout率
                 proj_drop_ratio=0.):
        super().__init__()
        self.heads = heads
        self.embed_dim = embed_dim
        head_dim = embed_dim // heads  # 每一个head的dim数
        self.scale = head_dim ** -0.5  # ViT-B 就是 768//12 = 64

        # 这里的q,k,v 可以用三个Linear层
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # 或者一个Linear层，但是out_channel为三倍，并行的思想
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        # self.softmax = nn.Softmax(dim = -1) # 对每一行进行softmax

        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # Multi-head的拼接，需要一个参数Wo，靠此层进行训练
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        # [batch_size, seq_len        , total_embed_dim]
        B, N, C = x.shape
        assert C == self.embed_dim

        # 1. qkv -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        #  # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
