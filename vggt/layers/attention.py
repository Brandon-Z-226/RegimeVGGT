import os
from pathlib import Path
import time
from PIL import Image
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from tqdm.std import tqdm
from methods.fastvggt_merge import (
    token_merge_bipartite2d,
)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

XFORMERS_AVAILABLE = False

# Global variables for attention visualization
vis_attn_map = False
current_images = []
attention_map = None


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        kv_group_size: int = 1,
        fused_attn: bool = True,
        rope=None,
        global_merging=None,
        patch_width: int = 37,
        patch_height: int = 28,
        merge_ratio: float = 0.9,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.fused_attn = fused_attn
        self.merge_ratio = merge_ratio

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.kv_group_size = kv_group_size

    def forward(
        self, x: Tensor, pos=None, global_merging=None,
        importance_scores=None, merge_cache_key=None, merge_alpha: float = 0.1,
    ) -> Tensor:
        merge_num = list(range(24))

        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if vis_attn_map and global_merging is not None:
            # s1 Chunk computation of q@k
            chunk_size = 4096
            attn_maps = []
            total_chunks = (
                q.size(-2) + chunk_size - 1
            ) // chunk_size  # Calculate total number of chunks
            start_time_total = time.time()
            with torch.no_grad():
                for start in tqdm(
                    range(0, q.size(-2), chunk_size),
                    total=total_chunks,
                    desc="Processing chunks",
                ):
                    end = min(start + chunk_size, q.size(-2))
                    q_chunk = q[:, :, start:end, :]  # (1, 16, chunk_size, 64)
                    start_time_chunk = time.time()
                    attn_chunk = q_chunk @ k.transpose(
                        -2, -1
                    )  # (1, 16, chunk_size, 34353)
                    attn_maps.append(attn_chunk.cpu())
                    end_time_chunk = time.time()
                    print(
                        f"Chunk {start}:{end} processed in {end_time_chunk - start_time_chunk:.4f} seconds"
                    )
            end_time_total = time.time()
            print(
                f"\nTotal processing time: {end_time_total - start_time_total:.4f} seconds"
            )

            attn_map = torch.cat(attn_maps, dim=-2)
            attn = attn_map[0].mean(0)
            frame_token_num = self.patch_height * self.patch_width + 5
            for target_token_idx in [
                0,
                self.patch_height * self.patch_width,
                self.patch_height * self.patch_width * 10,
            ]:  # Iterate through each image's target_token
                for image_idx in range(
                    len(current_images)
                ):  # Corresponding to which image to visualize
                    target_attn = attn[
                        target_token_idx,
                        image_idx * frame_token_num : (image_idx + 1) * frame_token_num,
                    ]
                    target_attn_map = target_attn[5:].reshape(
                        self.patch_height, self.patch_width
                    )
                    # 1) Read original image to get true size (H, W)
                    image_path = current_images[image_idx]
                    p = Path(image_path)
                    parts = p.parts
                    scene_name = parts[-4]

                    image = Image.open(image_path).convert("RGB")
                    img_width, img_height = image.size  # PIL size: (W, H)

                    # Upsample attention map to the original image size
                    target_attn_map = F.interpolate(
                        target_attn_map.unsqueeze(0).unsqueeze(0),  # (1,1,h,w)
                        size=(img_height, img_width),
                        mode="bilinear",
                    )
                    target_attn_map = target_attn_map.squeeze()

                    # Convert image to numpy for blending
                    img_np = np.array(image) / 255.0

                    # 2. Normalize attention map
                    target_attn_map = (target_attn_map - target_attn_map.min()) / (
                        target_attn_map.max() - target_attn_map.min()
                    )

                    # 3. Color attention map
                    cmap = plt.get_cmap("jet")
                    attn_color = cmap(target_attn_map.cpu().float().numpy())
                    attn_color = attn_color[:, :, :3]

                    # 4. Blend attention and original image
                    overlay = img_np * 0.5 + attn_color * 0.5

                    plt.imshow(overlay, cmap="viridis")
                    output_dir = f"attention_map/{scene_name}/block_{attention_map}/token_{target_token_idx}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"color_{image_idx}.png")
                    plt.savefig(output_path)
                    plt.close()

        if global_merging is not None and global_merging in merge_num:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(33)

            r = int(x.shape[1] * self.merge_ratio)

            m, u = token_merge_bipartite2d(
                x,
                self.patch_width,
                self.patch_height,
                2,
                2,
                r,
                False,
                generator,
                enable_protection=True,
                importance_scores=importance_scores,
                alpha=merge_alpha,
                cache_key=merge_cache_key,
            )

            m_a, u_a = (m, u)

            B_q, H_q, N_q, D_q = q.shape

            q_merge_in = q.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
            k_merge_in = k.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
            v_merge_in = v.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)

            q_out, k_out, v_out = m_a(
                q_merge_in,
                mode="mean",
                extra_tensors=k_merge_in,
                extra_tensors_2=v_merge_in,
            )

            del q_merge_in, k_merge_in, v_merge_in

            N_m = q_out.shape[1]
            q = q_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
            k = k_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
            v = v_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)

            del q_out, k_out, v_out

            N = N_m

        # ToMe size-aware bias: log(size) added to attention logits pre-softmax
        # so a merged dst representing r originals contributes ∝ r·v_dst
        # instead of being treated as a single token. Training-free.
        # When merge fell through to do_nothing (r=0), m_a has no .size and
        # bias is irrelevant (every K already represents 1 original token).
        # Ablation hatch: GEOVGGT_DISABLE_SIZE_BIAS=1 forces the unbiased path
        # so we can measure the +0.001 contribution. Default behaviour
        # (env unset) is identical to before — main experiments unaffected.
        import os as _os_sb
        _SB_DISABLED = _os_sb.environ.get("GEOVGGT_DISABLE_SIZE_BIAS") == "1"
        size_bias = None
        if (not _SB_DISABLED) and global_merging is not None and global_merging in merge_num:
            merge_size_attr = getattr(m_a, "size", None)
            if merge_size_attr is not None:
                # shape [N_kv]; broadcast to [B, H, N_q, N_kv] via the matmul path.
                size_bias = torch.log(merge_size_attr.clamp_min(1.0)).to(q.dtype)

        if size_bias is not None:
            # Encode log(size) bias into an extra K-dim so flash attention can
            # be used (sdpa with a float attn_mask falls off flash; mem_efficient
            # is ~50% slower than flash on long-sequence workloads).
            #
            # With Q_aug = [Q, 1, 0...0] and K_aug = [K, sqrt(d)*log(size), 0...0]
            # we have <Q_aug, K_aug>/sqrt(d) = <Q,K>/sqrt(d) + log(size_j),
            # exactly the additive ToMe bias. V is padded to match head_dim
            # (the trailing zero cols dot with zero attention contribution).
            #
            # Pad to next flash-supported head_dim (80 from base 64 in VGGT-1B).
            import math
            d_head = q.shape[-1]
            flash_supported = (8, 16, 32, 64, 80, 96, 112, 128, 160, 192, 224, 256)
            target_d = next((fd for fd in flash_supported if fd > d_head), d_head + 1)
            pad_extra = target_d - d_head - 1   # extra zero cols after the bias col
            sqrt_d = math.sqrt(d_head)

            log_size_scaled = (size_bias * sqrt_d).view(1, 1, -1, 1).expand(
                k.shape[0], k.shape[1], k.shape[2], 1
            )
            ones_q = torch.ones(*q.shape[:-1], 1, device=q.device, dtype=q.dtype)
            zero_v = torch.zeros(*v.shape[:-1], 1, device=v.device, dtype=v.dtype)

            cat_pieces = lambda *cols: torch.cat(cols, dim=-1)
            if pad_extra > 0:
                q_pad = torch.zeros(*q.shape[:-1], pad_extra, device=q.device, dtype=q.dtype)
                k_pad = torch.zeros(*k.shape[:-1], pad_extra, device=k.device, dtype=k.dtype)
                v_pad = torch.zeros(*v.shape[:-1], pad_extra, device=v.device, dtype=v.dtype)
                q_aug = cat_pieces(q, ones_q, q_pad)
                k_aug = cat_pieces(k, log_size_scaled, k_pad)
                v_aug = cat_pieces(v, zero_v, v_pad)
            else:
                q_aug = cat_pieces(q, ones_q)
                k_aug = cat_pieces(k, log_size_scaled)
                v_aug = cat_pieces(v, zero_v)

            x_aug = F.scaled_dot_product_attention(
                q_aug,
                k_aug,
                v_aug,
                scale=1.0 / sqrt_d,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            # Strip the bias-encoding/padding tail; V originals occupy first d_head cols.
            x = x_aug[..., :d_head]
            del q_aug, k_aug, v_aug, x_aug, log_size_scaled, ones_q, zero_v
        else:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        del q, k, v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if global_merging is not None and global_merging in merge_num:
            x = u_a(x)
        return x


class MemEffAttention(Attention):
    def forward(
        self, x: Tensor, attn_bias=None, pos=None, global_merging=None,
        importance_scores=None, merge_cache_key=None, merge_alpha: float = 0.1,
    ) -> Tensor:
        assert (
            pos is None or self.rope is not None
        ), "Position encoding is only supported with RoPE"
        # When global_merging is active, use base Attention.forward() which has merge logic
        if global_merging is not None:
            return super().forward(
                x, pos=pos, global_merging=global_merging,
                importance_scores=importance_scores,
                merge_cache_key=merge_cache_key,
                merge_alpha=merge_alpha,
            )
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(
                x, pos=pos, global_merging=global_merging,
                importance_scores=importance_scores,
                merge_cache_key=merge_cache_key,
                merge_alpha=merge_alpha,
            )

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = qkv.unbind(2)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Use scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
