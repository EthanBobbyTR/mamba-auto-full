"""
Autoresearch pretraining script - Mamba-2 (SSD). Single-GPU, single-file.
Pure PyTorch implementation of the Structured State Space Duality algorithm.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Mamba-2 Model
# ---------------------------------------------------------------------------

@dataclass
class Mamba2Config:
    d_model: int = 768
    n_layer: int = 16
    d_state: int = 128       # N: SSM state dimension
    d_conv: int = 4          # causal conv kernel size
    expand: int = 2          # expansion factor E
    head_dim: int = 64       # P: per-head dimension
    chunk_size: int = 64     # Q: SSD block length
    vocab_size: int = 8192
    max_seq_len: int = 2048

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.head_dim == 0
        self.n_heads = self.d_inner // self.head_dim


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ---------------------------------------------------------------------------
# SSD Algorithm (Structured State Space Duality)
# ---------------------------------------------------------------------------

def segsum(x):
    """Stable segment sum for 1-semiseparable matrix construction.
    Input: x shape (..., T)
    Output: shape (..., T, T) lower-triangular matrix
    Entry [i,j] = sum(x[j+1:i+1]) for j <= i, -inf otherwise
    """
    T = x.size(-1)
    # Expand for pairwise computation
    x = x.unsqueeze(-1).expand(*x.shape, T)              # (..., T, T)
    # Zero out upper triangle (keep strict lower triangle for cumsum)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    # Cumsum along rows gives segment sums
    x_segsum = torch.cumsum(x, dim=-2)
    # Mask out upper triangle in result (set to -inf so exp gives 0)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(X, A, B, C, chunk_size, initial_states=None):
    """
    Structured State Space Duality - core Mamba-2 algorithm.

    Args:
        X: (batch, length, n_heads, head_dim) -- input values
        A: (batch, length, n_heads) -- log decay rates (negative)
        B: (batch, length, n_heads, d_state) -- input-to-state
        C: (batch, length, n_heads, d_state) -- state-to-output
        chunk_size: int -- block length Q
        initial_states: optional (batch, n_heads, head_dim, d_state)

    Returns:
        Y: (batch, length, n_heads, head_dim)
        final_state: (batch, n_heads, head_dim, d_state)
    """
    batch, seqlen, n_heads, head_dim = X.shape
    d_state = B.shape[-1]
    assert seqlen % chunk_size == 0
    n_chunks = seqlen // chunk_size

    # Reshape into chunks: (batch, n_chunks, chunk_size, ...)
    X = X.reshape(batch, n_chunks, chunk_size, n_heads, head_dim)
    A = A.reshape(batch, n_chunks, chunk_size, n_heads)
    B = B.reshape(batch, n_chunks, chunk_size, n_heads, d_state)
    C = C.reshape(batch, n_chunks, chunk_size, n_heads, d_state)

    # Cumulative sum of A within each chunk (for decay computation)
    A_cumsum = torch.cumsum(A, dim=2)  # (batch, n_chunks, chunk_size, n_heads)

    # ===== Step 1: Intra-chunk (diagonal blocks) =====
    # Build decay matrix L within each chunk
    # L[i,j] = exp(cumA[i] - cumA[j]) for j <= i
    # Rearrange A to (batch, n_heads, n_chunks, chunk_size) for segsum
    A_for_segsum = A.permute(0, 3, 1, 2)  # (batch, n_heads, n_chunks, chunk_size)
    L = torch.exp(segsum(A_for_segsum))   # (batch, n_heads, n_chunks, chunk_size, chunk_size)

    # Compute intra-chunk output: Y_diag = (C @ B^T * L) @ X
    # Y_diag[b,c,l,h,p] = sum_s sum_n C[b,c,l,h,n] * B[b,c,s,h,n] * L[b,h,c,l,s] * X[b,c,s,h,p]
    Y_diag = torch.einsum(
        "bclhn, bcshn, bhcls, bcshp -> bclhp",
        C, B, L, X
    )

    # ===== Step 2: Chunk state computation =====
    # Accumulate input within each chunk into state representation
    # decay_states[i] = exp(cumA_end - cumA[i]) to weight each position's contribution
    decay_states = torch.exp(
        A_cumsum[:, :, -1:, :] - A_cumsum
    )  # (batch, n_chunks, chunk_size, n_heads)

    # states = sum_l B[l] * decay[l] * X[l] (outer product over n and p)
    states = torch.einsum(
        "bclhn, bcl h, bclhp -> bchpn",
        B,
        decay_states,
        X,
    )  # (batch, n_chunks, n_heads, head_dim, d_state)

    # ===== Step 3: Inter-chunk SSM recurrence =====
    # Propagate states across chunks using cumulative decay
    # First, build the chunk-level decay factors
    chunk_decay = A_cumsum[:, :, -1, :]  # (batch, n_chunks, n_heads)

    # Pad with zero at the start for initial state contribution
    chunk_decay_padded = F.pad(chunk_decay, (0, 0, 1, 0))  # (batch, n_chunks+1, n_heads)
    chunk_decay_padded = chunk_decay_padded.permute(0, 2, 1)  # (batch, n_heads, n_chunks+1)
    decay_chunk = torch.exp(segsum(chunk_decay_padded))  # (batch, n_heads, n_chunks+1, n_chunks+1)

    # Include initial states if provided
    if initial_states is None:
        initial_states = torch.zeros(
            batch, n_heads, head_dim, d_state,
            device=X.device, dtype=X.dtype
        )
    # Prepend initial states to chunk states
    states_with_init = torch.cat([
        initial_states.unsqueeze(1),  # (batch, 1, n_heads, head_dim, d_state)
        states
    ], dim=1)  # (batch, n_chunks+1, n_heads, head_dim, d_state)

    # Apply chunk-level recurrence: new_states[z] = sum_c decay[z,c] * states[c]
    new_states = torch.einsum(
        "bhzc, bchpn -> bzhpn",
        decay_chunk,
        states_with_init,
    )  # (batch, n_chunks+1, n_heads, head_dim, d_state)

    # Take states 1..n_chunks (corresponding to each chunk's accumulated state from prior chunks)
    states_for_output = new_states[:, 1:]  # (batch, n_chunks, n_heads, head_dim, d_state)
    final_state = new_states[:, -1]        # (batch, n_heads, head_dim, d_state)

    # ===== Step 4: State-to-output (off-diagonal contribution) =====
    # Convert accumulated states to output contributions within each chunk
    state_decay_out = torch.exp(A_cumsum)  # (batch, n_chunks, chunk_size, n_heads)

    Y_off = torch.einsum(
        "bclhn, bchpn, bclh -> bclhp",
        C,
        states_for_output,
        state_decay_out,
    )

    # Combine intra-chunk and inter-chunk contributions
    Y = Y_diag + Y_off

    # Reshape back to (batch, seqlen, n_heads, head_dim)
    Y = Y.reshape(batch, seqlen, n_heads, head_dim)

    return Y, final_state


# ---------------------------------------------------------------------------
# Mamba-2 Block
# ---------------------------------------------------------------------------

class Mamba2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_inner
        n = config.d_state
        h = config.n_heads

        # Input projection: x -> (z, xBC, dt)
        # z: gate (d_inner), xBC: conv input (d_inner + 2*d_state), dt: time step (n_heads)
        d_in_proj = 2 * d + 2 * n + h
        self.in_proj = nn.Linear(config.d_model, d_in_proj, bias=False)

        # Causal 1D convolution (depthwise on xBC)
        conv_dim = d + 2 * n
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=config.d_conv, groups=conv_dim,
            padding=config.d_conv - 1,
        )

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.empty(h))
        self.A_log = nn.Parameter(torch.empty(h))
        self.D = nn.Parameter(torch.empty(h))

        # Output: gated norm + projection
        self.norm = RMSNorm(d)
        self.out_proj = nn.Linear(d, config.d_model, bias=False)

    def forward(self, u):
        """
        u: (batch, seqlen, d_model)
        returns: (batch, seqlen, d_model)
        """
        config = self.config
        batch, seqlen, _ = u.shape
        chunk_size = config.chunk_size

        # Pad sequence to multiple of chunk_size
        pad_len = (chunk_size - seqlen % chunk_size) % chunk_size
        if pad_len > 0:
            u = F.pad(u, (0, 0, 0, pad_len))
        padded_len = u.shape[1]

        # Input projection
        zxbcdt = self.in_proj(u)

        # Split into components
        d = config.d_inner
        n = config.d_state
        h = config.n_heads

        z = zxbcdt[:, :, :d]                          # (batch, L, d_inner) - gate
        xBC = zxbcdt[:, :, d:2*d + 2*n]               # (batch, L, d_inner + 2*d_state) - conv input
        dt = zxbcdt[:, :, 2*d + 2*n:]                 # (batch, L, n_heads) - time steps

        # Softplus for positive time steps
        dt = F.softplus(dt + self.dt_bias)  # (batch, L, n_heads)

        # Causal 1D convolution + activation
        xBC = self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :padded_len]
        xBC = F.silu(xBC)

        # Split convolved output into x, B, C
        x = xBC[:, :, :d]                             # (batch, L, d_inner)
        B = xBC[:, :, d:d+n]                          # (batch, L, d_state)
        C = xBC[:, :, d+n:d+2*n]                      # (batch, L, d_state)

        # Reshape x for SSD: (batch, L, n_heads, head_dim)
        x = x.reshape(batch, padded_len, h, config.head_dim)

        # Discretized A (log space, always negative)
        A = -torch.exp(self.A_log)  # (n_heads,)

        # Build per-position A: A * dt
        A = A.unsqueeze(0).unsqueeze(0) * dt  # (batch, L, n_heads)

        # Expand B, C to have n_heads dim: (batch, L, n_heads, d_state)
        # B and C are shared across heads
        B = B.unsqueeze(2).expand(-1, -1, h, -1)
        C = C.unsqueeze(2).expand(-1, -1, h, -1)

        # Scale x by dt for discretization
        x_scaled = x * dt.unsqueeze(-1)

        # Run SSD
        y, _ = ssd(x_scaled, A, B, C, chunk_size)
        # y: (batch, L, n_heads, head_dim)

        # Skip connection with D parameter
        y = y + x * self.D.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Reshape back to (batch, L, d_inner)
        y = y.reshape(batch, padded_len, d)

        # Gated output: y * SiLU(z), then norm, then project
        y = y * F.silu(z)
        y = self.norm(y)
        y = self.out_proj(y)

        # Remove padding
        if pad_len > 0:
            y = y[:, :seqlen]

        return y


# ---------------------------------------------------------------------------
# Mamba-2 Language Model
# ---------------------------------------------------------------------------

class Mamba2LM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mixer': Mamba2Block(config),
                'norm': RMSNorm(config.d_model),
            })
            for _ in range(config.n_layer)
        ])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        config = self.config
        s = 3**0.5 * config.d_model**-0.5

        # Embedding and unembedding
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        for layer in self.layers:
            mixer = layer['mixer']
            # Linear projections
            nn.init.uniform_(mixer.in_proj.weight, -s, s)
            nn.init.zeros_(mixer.out_proj.weight)
            # Conv1d: default PyTorch init is fine
            # A_log: initialize for decay rates between 1 and 16
            nn.init.uniform_(mixer.A_log, math.log(1), math.log(16))
            # dt_bias: initialize so dt starts in reasonable range [0.001, 0.1]
            dt_init = torch.exp(
                torch.empty(config.n_heads).uniform_(math.log(0.001), math.log(0.1))
            )
            mixer.dt_bias.copy_(torch.log(torch.exp(dt_init) - 1))  # inverse softplus
            # D: skip connection, start at 1
            mixer.D.fill_(1.0)
            # RMSNorm: ones (already default)

        # Cast embedding to bf16 for efficiency
        self.embedding.to(dtype=torch.bfloat16)

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward = 3x forward)."""
        config = self.config
        d = config.d_model
        d_inner = config.d_inner
        n = config.d_state
        h = config.n_heads
        p = config.head_dim
        Q = config.chunk_size

        per_layer = 0
        # in_proj: d_model -> (2*d_inner + 2*d_state + n_heads)
        proj_out = 2 * d_inner + 2 * n + h
        per_layer += 2 * d * proj_out
        # conv1d: depthwise, kernel_size * channels
        per_layer += 2 * (d_inner + 2 * n) * config.d_conv
        # SSD: approximate as O(chunk_size * d_state * n_heads * head_dim) amortized
        per_layer += 2 * Q * n * h * p / Q  # simplifies to 2 * n * h * p
        # Plus the quadratic attention within chunks: O(chunk_size * n_heads * head_dim)
        per_layer += 2 * Q * h * p
        # out_proj: d_inner -> d_model
        per_layer += 2 * d_inner * d
        # Total for all layers
        total_flops = config.n_layer * per_layer
        # Embedding lookup is negligible, lm_head projection
        total_flops += 2 * d * config.vocab_size
        # Factor of 3 for forward + backward
        return int(total_flops * 3)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def setup_optimizer(self, lr=0.001, embed_lr=0.1, unembed_lr=0.004,
                        scalar_lr=0.01, weight_decay=0.1, betas=(0.9, 0.95)):
        """Setup AdamW with per-group learning rates."""
        d = self.config.d_model
        dmodel_lr_scale = (d / 768) ** -0.5
        print(f"Scaling LRs by 1/sqrt({d}/768) = {dmodel_lr_scale:.6f}")

        # Collect parameters by type
        embed_params = list(self.embedding.parameters())
        lm_head_params = list(self.lm_head.parameters())
        matrix_params = []
        scalar_params = []

        for layer in self.layers:
            mixer = layer['mixer']
            # Matrix params: in_proj, out_proj, conv1d
            matrix_params.extend([mixer.in_proj.weight, mixer.out_proj.weight])
            matrix_params.append(mixer.conv1d.weight)
            if mixer.conv1d.bias is not None:
                scalar_params.append(mixer.conv1d.bias)
            # Scalar params: A_log, dt_bias, D, norm weights
            scalar_params.extend([mixer.A_log, mixer.dt_bias, mixer.D])
            scalar_params.append(mixer.norm.weight)
            scalar_params.append(layer['norm'].weight)
        scalar_params.append(self.norm_f.weight)

        # Verify all params are accounted for
        all_params = set(self.parameters())
        grouped = set(embed_params + lm_head_params + matrix_params + scalar_params)
        assert all_params == grouped, f"Missing {len(all_params - grouped)} params"

        param_groups = [
            dict(params=lm_head_params, lr=unembed_lr * dmodel_lr_scale,
                 betas=betas, eps=1e-10, weight_decay=0.0),
            dict(params=embed_params, lr=embed_lr * dmodel_lr_scale,
                 betas=betas, eps=1e-10, weight_decay=0.0),
            dict(params=matrix_params, lr=lr * dmodel_lr_scale,
                 betas=betas, eps=1e-10, weight_decay=weight_decay),
            dict(params=scalar_params, lr=scalar_lr * dmodel_lr_scale,
                 betas=betas, eps=1e-10, weight_decay=0.0),
        ]
        optimizer = torch.optim.AdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        x = self.embedding(idx)
        x = F.rms_norm(x, (x.size(-1),))  # pre-norm on embeddings

        for layer in self.layers:
            x = x + layer['mixer'](layer['norm'](x))

        x = self.norm_f(x)
        logits = self.lm_head(x).float()

        # Logit softcapping
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=reduction
            )
            return loss
        return logits

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = 16              # number of Mamba-2 layers
D_MODEL = 768           # model dimension
EXPAND = 2              # expansion factor (d_inner = EXPAND * D_MODEL)
HEAD_DIM = 64           # SSM head dimension
D_STATE = 64            # SSM state dimension
D_CONV = 4              # causal conv kernel size
CHUNK_SIZE = 64         # SSD block length

# Optimization
TOTAL_BATCH_SIZE = 2**17  # ~131K tokens per optimizer step
LEARNING_RATE = 0.001     # base LR for projection matrices
EMBED_LR = 0.1           # embedding LR
UNEMBED_LR = 0.004       # lm_head LR
SCALAR_LR = 0.01         # scalar params LR (A_log, dt_bias, D, norms)
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.02      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.4     # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0      # final LR as fraction of initial

# Device
DEVICE_BATCH_SIZE = 4     # micro-batch size (fits in 24GB A10G)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
A10G_BF16_PEAK_FLOPS = 70.0e12  # NVIDIA A10G ~70 TFLOPS bf16

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

config = Mamba2Config(
    d_model=D_MODEL,
    n_layer=DEPTH,
    d_state=D_STATE,
    d_conv=D_CONV,
    expand=EXPAND,
    head_dim=HEAD_DIM,
    chunk_size=CHUNK_SIZE,
    vocab_size=vocab_size,
    max_seq_len=MAX_SEQ_LEN,
)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = Mamba2LM(config)
model.to_empty(device=device)
model.init_weights()

num_params = model.num_params()
num_flops_per_token = model.estimate_flops()
print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0, \
    f"TOTAL_BATCH_SIZE ({TOTAL_BATCH_SIZE}) must be divisible by tokens_per_fwdbwd ({tokens_per_fwdbwd})"
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    lr=LEARNING_RATE,
    embed_lr=EMBED_LR,
    unembed_lr=UNEMBED_LR,
    scalar_lr=SCALAR_LR,
    weight_decay=WEIGHT_DECAY,
    betas=ADAM_BETAS,
)

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / A10G_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up -- but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / A10G_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
