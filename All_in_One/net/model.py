import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

from .DINOv3_utils import DINOExtractor, SemanticAdapter, SemanticModulator


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class DegradationRepresentationBlock(nn.Module):
    """
    Extracts global and local degradation representations.

    Returns:
        z_g: Global degradation vector with shape [B, degradation_dim].
        local_tokens: Local degradation tokens with shape [B, K, degradation_dim],
            where K = local_token_grid * local_token_grid.
    """
    def __init__(self, in_channels=3, mid_channels=64, degradation_dim=512, local_token_grid=2):
        super().__init__()
        act = nn.LeakyReLU(0.1, inplace=True)
        self.local_token_grid = local_token_grid

        self.res = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True), act,
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        )
        self.res_skip = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True), act,
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(mid_channels * 2, mid_channels, bias=True), act,
            nn.Linear(mid_channels, degradation_dim, bias=True)
        )
        self.global_norm = nn.LayerNorm(degradation_dim)

        self.local_mlp = nn.Sequential(
            nn.Linear(mid_channels * 2, mid_channels, bias=True), act,
            nn.Linear(mid_channels, degradation_dim, bias=True)
        )
        self.local_norm = nn.LayerNorm(degradation_dim)

    @staticmethod
    def _region_stats(f, grid_size):
        pooled_mean = F.adaptive_avg_pool2d(f, (grid_size, grid_size))
        pooled_sq_mean = F.adaptive_avg_pool2d(f * f, (grid_size, grid_size))
        pooled_var = torch.clamp(pooled_sq_mean - pooled_mean * pooled_mean, min=0.0)
        pooled_std = torch.sqrt(pooled_var + 1e-6)

        mu = rearrange(pooled_mean, 'b c gh gw -> b (gh gw) c')
        std = rearrange(pooled_std, 'b c gh gw -> b (gh gw) c')
        return torch.cat([mu, std], dim=-1)

    def forward(self, x):
        f = self.conv(self.res(x) + self.res_skip(x))

        mu = f.mean(dim=(2, 3))
        std = torch.sqrt(f.var(dim=(2, 3), unbiased=False) + 1e-6)

        z_g = self.global_mlp(torch.cat([mu, std], dim=1))
        z_g = self.global_norm(z_g)
        z_g = z_g / (z_g.norm(dim=1, keepdim=True) + 1e-6)

        local_tokens = self.local_mlp(self._region_stats(f, self.local_token_grid))
        local_tokens = self.local_norm(local_tokens)

        return z_g, local_tokens


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, 3, 1, 1,
            groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, dino_dim=256):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.sft = SemanticModulator(dim, condition_dim=dino_dim)

    def forward(self, x, dino_feat):
        x = x + self.attn(self.norm1(x))
        x = self.sft(x, dino_feat)
        return x + self.ffn(self.norm2(x))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, degradation_dim, num_degradation_tokens=16):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.dim = dim
        self.num_degradation_tokens = num_degradation_tokens
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.deg_tokens = nn.Parameter(torch.randn(1, num_degradation_tokens, dim))
        self.token_scale = nn.Linear(degradation_dim, dim, bias=True)
        self.token_shift = nn.Linear(degradation_dim, dim, bias=True)
        self.local_proj = nn.Linear(degradation_dim, dim, bias=True)
        self.token_norm = nn.LayerNorm(dim)

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x, degradation_vector, local_tokens):
        b, c, h, w = x.shape

        base_tokens = self.deg_tokens.expand(b, -1, -1)
        scale = self.token_scale(degradation_vector).unsqueeze(1)
        shift = self.token_shift(degradation_vector).unsqueeze(1)
        global_tokens = base_tokens * (1 + scale) + shift

        projected_local_tokens = self.local_proj(local_tokens)
        memory_tokens = torch.cat([global_tokens, projected_local_tokens], dim=1)
        memory_tokens = self.token_norm(memory_tokens)

        q = self.q_dwconv(self.q_proj(x))
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        k = self.k_proj(memory_tokens)
        v = self.v_proj(memory_tokens)
        k = rearrange(k, 'b t (head c) -> b head t c', head=self.num_heads)
        v = rearrange(v, 'b t (head c) -> b head t c', head=self.num_heads)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head (h w) c -> b (head c) h w', h=h, w=w)
        return self.project_out(out)


class TransformerStack(nn.Module):
    def __init__(
        self, num_blocks, dim, num_heads, ffn_expansion_factor,
        bias=False, LayerNorm_type='WithBias', dino_dim=256
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, dino_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x, dino_feat):
        for blk in self.blocks:
            x = blk(x, dino_feat=dino_feat)
        return x


class HQSXUpdate(nn.Module):
    """
    Performs the x-update step in the HQS-inspired optimization process.
    The update consists of a degradation-aware data term and a coupling term.
    """
    def __init__(self, dim, degradation_dim=512, num_heads=2, bias=False, num_degradation_tokens=16):
        super().__init__()
        self.phi = CrossAttention(dim, num_heads, bias, degradation_dim, num_degradation_tokens)
        self.phit = Attention(dim, num_heads, bias)
        self.log_tau = nn.Parameter(torch.zeros(1))
        self.log_mu = nn.Parameter(torch.zeros(1))

    def forward(self, x, v, obs, degradation_vector, local_tokens):
        tau = F.softplus(self.log_tau) + 1e-6
        mu = F.softplus(self.log_mu) + 1e-6

        residual = self.phi(x, degradation_vector, local_tokens) - obs
        grad_data = self.phit(residual)
        grad_couple = x - v

        return x - tau * (grad_data + mu * grad_couple)


class HQSBaseBlock(nn.Module):
    """
    One HQS-inspired restoration block.

    It alternates between:
        1. x-update: degradation-aware observation consistency.
        2. v-update: visual-guided prior refinement.
    """
    def __init__(
        self, dim, num_blocks, num_heads, ffn_expansion_factor,
        degradation_dim=512, bias=False, LayerNorm_type='WithBias',
        num_degradation_tokens=16, dino_dim=256
    ):
        super().__init__()
        self.x_update = HQSXUpdate(
            dim=dim,
            degradation_dim=degradation_dim,
            num_heads=max(1, num_heads),
            bias=bias,
            num_degradation_tokens=num_degradation_tokens
        )
        self.denoiser = TransformerStack(
            num_blocks=num_blocks,
            dim=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dino_dim=dino_dim
        )
        self.log_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, v, obs, degradation_vector, local_tokens, dino_feat):
        x = self.x_update(x, v, obs, degradation_vector, local_tokens)

        alpha = F.softplus(self.log_alpha) + 1e-6
        v_hat = self.denoiser(x, dino_feat=dino_feat)
        v = v + alpha * (v_hat - v)

        return x, v


class DVANet(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        degradation_dim=512,
        drb_mid_channels=64,
        local_token_grid=2,
        num_degradation_tokens=16,
        dino_weight_path="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        dino_extract_ids=(5, 11, 17, 23),
        dino_adapter_out_dim=256,
        dino_adapter_sizes=(32, 16, 8, 4),
        dino_bottleneck=64,
        device="cuda",
    ):
        super().__init__()
        self.dino_adapter_sizes = dino_adapter_sizes

        self.dino_extractor = DINOExtractor(
            weights_path=dino_weight_path,
            extract_ids=list(dino_extract_ids),
            device=device
        )
        self.dino_adapter = SemanticAdapter(
            in_dim=1024,
            out_dim=dino_adapter_out_dim,
            sizes=dino_adapter_sizes,
            bottleneck=dino_bottleneck
        )
        for p in self.dino_extractor.parameters():
            p.requires_grad = False

        self.drb = DegradationRepresentationBlock(
            in_channels=inp_channels,
            mid_channels=drb_mid_channels,
            degradation_dim=degradation_dim,
            local_token_grid=local_token_grid
        )

        self.patch_embed_x = OverlapPatchEmbed(inp_channels, dim, bias=bias)
        self.patch_embed_v = OverlapPatchEmbed(inp_channels, dim, bias=bias)
        self.obs_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        self.encoder_level1 = HQSBaseBlock(
            dim=dim, num_blocks=num_blocks[0], num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor, degradation_dim=degradation_dim,
            bias=bias, LayerNorm_type=LayerNorm_type, num_degradation_tokens=num_degradation_tokens,
            dino_dim=dino_adapter_out_dim
        )
        self.down1_2_x = Downsample(dim)
        self.down1_2_v = Downsample(dim)
        self.down1_2_obs = Downsample(dim)

        self.encoder_level2 = HQSBaseBlock(
            dim=int(dim * 2), num_blocks=num_blocks[1], num_heads=heads[1],
            ffn_expansion_factor=ffn_expansion_factor, degradation_dim=degradation_dim,
            bias=bias, LayerNorm_type=LayerNorm_type, num_degradation_tokens=num_degradation_tokens,
            dino_dim=dino_adapter_out_dim
        )
        self.down2_3_x = Downsample(int(dim * 2))
        self.down2_3_v = Downsample(int(dim * 2))
        self.down2_3_obs = Downsample(int(dim * 2))

        self.encoder_level3 = HQSBaseBlock(
            dim=int(dim * 4), num_blocks=num_blocks[2], num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor, degradation_dim=degradation_dim,
            bias=bias, LayerNorm_type=LayerNorm_type, num_degradation_tokens=num_degradation_tokens,
            dino_dim=dino_adapter_out_dim
        )
        self.down3_4_x = Downsample(int(dim * 4))
        self.down3_4_v = Downsample(int(dim * 4))
        self.down3_4_obs = Downsample(int(dim * 4))

        self.latent = HQSBaseBlock(
            dim=int(dim * 8), num_blocks=num_blocks[3], num_heads=heads[3],
            ffn_expansion_factor=ffn_expansion_factor, degradation_dim=degradation_dim,
            bias=bias, LayerNorm_type=LayerNorm_type, num_degradation_tokens=num_degradation_tokens,
            dino_dim=dino_adapter_out_dim
        )

        self.up4_3_x = Upsample(int(dim * 8))
        self.up4_3_v = Upsample(int(dim * 8))
        self.up4_3_obs = Upsample(int(dim * 8))
        self.reduce_chan_level3_x = nn.Conv2d(int(dim * 8), int(dim * 4), 1, bias=bias)
        self.reduce_chan_level3_v = nn.Conv2d(int(dim * 8), int(dim * 4), 1, bias=bias)
        self.reduce_chan_level3_obs = nn.Conv2d(int(dim * 8), int(dim * 4), 1, bias=bias)

        self.decoder_level3 = HQSBaseBlock(
            dim=int(dim * 4), num_blocks=num_blocks[2], num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor, degradation_dim=degradation_dim,
            bias=bias, LayerNorm_type=LayerNorm_type, num_degradation_tokens=num_degradation_tokens,
            dino_dim=dino_adapter_out_dim
        )

        self.up3_2_x = Upsample(int(dim * 4))
        self.up3_2_v = Upsample(int(dim * 4))
        self.up3_2_obs = Upsample(int(dim * 4))
        self.reduce_chan_level2_x = nn.Conv2d(int(dim * 4), int(dim * 2), 1, bias=bias)
        self.reduce_chan_level2_v = nn.Conv2d(int(dim * 4), int(dim * 2), 1, bias=bias)
        self.reduce_chan_level2_obs = nn.Conv2d(int(dim * 4), int(dim * 2), 1, bias=bias)

        self.decoder_level2 = HQSBaseBlock(
            dim=int(dim * 2), num_blocks=num_blocks[1], num_heads=heads[1],
            ffn_expansion_factor=ffn_expansion_factor, degradation_dim=degradation_dim,
            bias=bias, LayerNorm_type=LayerNorm_type, num_degradation_tokens=num_degradation_tokens,
            dino_dim=dino_adapter_out_dim
        )

        self.up2_1_x = Upsample(int(dim * 2))
        self.up2_1_v = Upsample(int(dim * 2))
        self.up2_1_obs = Upsample(int(dim * 2))

        self.decoder_level1 = HQSBaseBlock(
            dim=int(dim * 2), num_blocks=num_blocks[0], num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor, degradation_dim=degradation_dim,
            bias=bias, LayerNorm_type=LayerNorm_type, num_degradation_tokens=num_degradation_tokens,
            dino_dim=dino_adapter_out_dim
        )

        self.refinement = HQSBaseBlock(
            dim=int(dim * 2), num_blocks=num_refinement_blocks, num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor, degradation_dim=degradation_dim,
            bias=bias, LayerNorm_type=LayerNorm_type, num_degradation_tokens=num_degradation_tokens,
            dino_dim=dino_adapter_out_dim
        )

        self.output = nn.Conv2d(int(dim * 2), out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img):
        with torch.no_grad():
            dino_feats = self.dino_extractor(inp_img)
        dino_feats = self.dino_adapter(dino_feats)

        degradation_vector, local_tokens = self.drb(inp_img)

        x1 = self.patch_embed_x(inp_img)
        v1 = self.patch_embed_v(inp_img)
        obs1 = self.obs_embed(inp_img)

        dino_l1 = dino_feats[0]
        x1, v1 = self.encoder_level1(x1, v1, obs1, degradation_vector, local_tokens, dino_feat=dino_l1)

        x2 = self.down1_2_x(x1)
        v2 = self.down1_2_v(v1)
        obs2 = self.down1_2_obs(obs1)

        dino_l2 = dino_feats[1]
        x2, v2 = self.encoder_level2(x2, v2, obs2, degradation_vector, local_tokens, dino_feat=dino_l2)

        x3 = self.down2_3_x(x2)
        v3 = self.down2_3_v(v2)
        obs3 = self.down2_3_obs(obs2)

        dino_l3 = dino_feats[2]
        x3, v3 = self.encoder_level3(x3, v3, obs3, degradation_vector, local_tokens, dino_feat=dino_l3)

        x4 = self.down3_4_x(x3)
        v4 = self.down3_4_v(v3)
        obs4 = self.down3_4_obs(obs3)

        dino_l4 = dino_feats[3]
        x4, v4 = self.latent(x4, v4, obs4, degradation_vector, local_tokens, dino_feat=dino_l4)

        dx3 = self.up4_3_x(x4)
        dv3 = self.up4_3_v(v4)
        dobs3 = self.up4_3_obs(obs4)

        dx3 = self.reduce_chan_level3_x(torch.cat([dx3, x3], dim=1))
        dv3 = self.reduce_chan_level3_v(torch.cat([dv3, v3], dim=1))
        dobs3 = self.reduce_chan_level3_obs(torch.cat([dobs3, obs3], dim=1))

        dx3, dv3 = self.decoder_level3(dx3, dv3, dobs3, degradation_vector, local_tokens, dino_feat=dino_l3)

        dx2 = self.up3_2_x(dx3)
        dv2 = self.up3_2_v(dv3)
        dobs2 = self.up3_2_obs(dobs3)

        dx2 = self.reduce_chan_level2_x(torch.cat([dx2, x2], dim=1))
        dv2 = self.reduce_chan_level2_v(torch.cat([dv2, v2], dim=1))
        dobs2 = self.reduce_chan_level2_obs(torch.cat([dobs2, obs2], dim=1))

        dx2, dv2 = self.decoder_level2(dx2, dv2, dobs2, degradation_vector, local_tokens, dino_feat=dino_l2)

        dx1 = self.up2_1_x(dx2)
        dv1 = self.up2_1_v(dv2)
        dobs1 = self.up2_1_obs(dobs2)

        dx1 = torch.cat([dx1, x1], dim=1)
        dv1 = torch.cat([dv1, v1], dim=1)
        dobs1 = torch.cat([dobs1, obs1], dim=1)

        dx1, dv1 = self.decoder_level1(dx1, dv1, dobs1, degradation_vector, local_tokens, dino_feat=dino_l1)

        _, out_prior = self.refinement(dx1, dv1, dobs1, degradation_vector, local_tokens, dino_feat=dino_l1)

        out = self.output(out_prior) + inp_img
        return out