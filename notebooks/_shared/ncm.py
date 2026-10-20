"""Small grid adaptation of Next-Concept Modeling (arXiv:2610.14162).

Independent PyTorch implementation of paper §3, not the released text checkpoint.
Two-dimensional scales/positions replace text lengths/positions. Target levels
can be packed with a block-diagonal attention mask or evaluated separately with
the equivalent objective. Conditional prefix tokens see only the prefix, while
targets see the prefix and their own level.
"""

from dataclasses import asdict, dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F


PAPER = "https://arxiv.org/abs/2610.14162"
REFERENCE = "https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203"


@dataclass(frozen=True)
class NCMConfig:
    size: int
    classes: int
    scales: tuple[int, ...]
    conditional: bool = False
    latent_dim: int = 64
    codebook_size: int = 64
    decoder_layers: int = 1
    width: int = 256
    layers: int = 5
    heads: int = 4
    dropout: float = 0.1
    commitment: float = 0.25
    nested_dropout: float = 0.5
    ema_decay: float = 0.99
    corruption: float = 0.1
    alpha: float = 1.0

    def __post_init__(self):
        if (not self.scales or self.scales[0] != 1 or self.scales[-1] != self.size
                or any(a >= b for a, b in zip(self.scales, self.scales[1:]))):
            raise ValueError("Scales must increase from 1 to the training size.")
        if self.width % self.heads or self.latent_dim % self.heads or self.latent_dim % 4:
            raise ValueError("Widths must be divisible by heads; latent_dim by four.")
        if not (0 <= self.corruption < 1 and 0 <= self.nested_dropout <= 1):
            raise ValueError("Invalid corruption/dropout probability.")

    def to_dict(self):
        return asdict(self)

    def sizes(self, size=None):
        # Reuse exactly the same number of codebooks at the transfer resolution.
        size = self.size if size is None else int(size)
        values = tuple(max(1, round(s * size / self.size)) for s in self.scales)
        values = (1, *values[1:-1], size)
        if any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError("This resolution collapses two trained scales.")
        return values


def positions(size, dim, device, dtype=torch.float32):
    """Resolution-independent 2D Fourier coordinates, returned as (1,H*W,D)."""
    axis = torch.linspace(-1, 1, size, device=device)
    rr, cc = torch.meshgrid(axis, axis, indexing="ij")
    frequency = torch.exp(torch.linspace(0, math.log(1000), dim // 4, device=device))
    r, c = rr.flatten()[:, None] * frequency, cc.flatten()[:, None] * frequency
    return torch.cat((r.sin(), r.cos(), c.sin(), c.cos()), -1)[None].to(dtype)


def resize(x, size):
    if x.shape[-2:] == (size, size):
        return x
    if size < x.shape[-1]:
        return F.adaptive_avg_pool2d(x, (size, size))
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


class AttentionBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width), nn.Dropout(dropout))

    def forward(self, x, prefix=0, allowed_mask=None):
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # QK RMS normalization, as in the official Sudoku configuration.
        q = F.rms_norm(q, (d // self.heads,))
        k = F.rms_norm(k, (d // self.heads,))
        mask = allowed_mask
        if prefix and mask is None:
            mask = torch.ones(n, n, device=x.device, dtype=torch.bool)
            mask[:prefix, prefix:] = False
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                          dropout_p=self.dropout if self.training else 0.)
        h = h.transpose(1, 2).reshape(b, n, d)
        x = x + F.dropout(self.projection(h), self.dropout, self.training)
        return x + self.mlp(self.norm2(x))


class LearnedCoarseResize(nn.Module):
    """Learned first-level down/up cascade, initialized as pooling/repetition.

    At 64x64 the trained 32x32 global operator is applied through interpolation;
    this is explicit out-of-distribution transfer, without new parameters.
    """
    def __init__(self, size, dim):
        super().__init__()
        self.size = size
        strides, remaining = [], size
        while remaining > 1:
            stride = 2 if remaining % 2 == 0 else remaining
            strides.append(stride)
            remaining //= stride
        self.down = nn.ModuleList(nn.Conv2d(dim, dim, s, stride=s) for s in strides)
        self.up = nn.ModuleList(nn.ConvTranspose2d(dim, dim, s, stride=s)
                                for s in reversed(strides))
        with torch.no_grad():
            indices = torch.arange(dim)
            for layer in self.down:
                layer.weight.zero_()
                layer.weight[indices, indices] = 1 / layer.kernel_size[0] ** 2
                layer.bias.zero_()
            for layer in self.up:
                layer.weight.zero_()
                layer.weight[indices, indices] = 1
                layer.bias.zero_()

    def encode(self, z):
        z = resize(z, self.size)
        for layer in self.down:
            z = layer(z)
        return z

    def decode(self, z, size):
        for layer in self.up:
            z = layer(z)
        return resize(z, size)


class EMACodebook(nn.Module):
    def __init__(self, entries, dim, decay):
        super().__init__()
        # Count EMA-learned code vectors in parameter totals, although Adam does
        # not optimize them. No uncounted trainable tokenizer is hidden here.
        self.embedding = nn.Embedding(entries, dim)
        self.embedding.weight.requires_grad_(False)
        self.decay = decay
        self.register_buffer("counts", torch.ones(entries))
        self.register_buffer("sums", self.embedding.weight.detach().clone())
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("updates", torch.tensor(0))

    @torch.no_grad()
    def nearest(self, vectors, update=False):
        flat = vectors.permute(0, 2, 3, 1).reshape(-1, vectors.shape[1]).float()
        entries = self.embedding.num_embeddings
        if update and not bool(self.initialized):
            # Data initialization avoids the huge first EMA jump from empty bins.
            centers = flat[torch.randint(len(flat), (entries,), device=flat.device)].clone()
            centers.add_(torch.randn_like(centers) * 1e-3)
            self.embedding.weight.copy_(centers)
            self.sums.copy_(centers)
            self.counts.fill_(1)
            self.initialized.fill_(True)
        weight = self.embedding.weight.float()
        distances = flat.square().sum(1, keepdim=True) + weight.square().sum(1) - 2 * flat @ weight.T
        ids = distances.argmin(1)
        if update:
            counts = torch.bincount(ids, minlength=entries).float()
            sums = torch.zeros_like(weight).index_add_(0, ids, flat)
            self.counts.lerp_(counts, 1 - self.decay)
            self.sums.lerp_(sums, 1 - self.decay)
            self.embedding.weight.copy_(self.sums / self.counts.clamp_min(1e-5)[:, None])
            self.updates.add_(1)
            if int(self.updates) % 100 == 0:
                dead = self.counts < .1
                replacements = flat[torch.randint(len(flat), (entries,), device=flat.device)]
                self.embedding.weight[dead] = replacements[dead]
                self.sums[dead], self.counts[dead] = replacements[dead], 1.
        return ids.view(vectors.shape[0], *vectors.shape[-2:])

    def lookup(self, ids):
        return self.embedding(ids).permute(0, 3, 1, 2)


class GridTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.latent_dim
        self.embedding = nn.Embedding(config.classes, d)
        self.pre_norm = nn.GroupNorm(8, d, affine=False)
        self.coarse = LearnedCoarseResize(config.size, d)
        self.books = nn.ModuleList(EMACodebook(config.codebook_size, d, config.ema_decay)
                                   for _ in config.scales)
        self.phi = nn.ModuleList(nn.Conv2d(d, d, 3, padding=1) for _ in config.scales)
        self.decoder = nn.ModuleList(AttentionBlock(d, config.heads, config.dropout)
                                     for _ in range(config.decoder_layers))
        self.decoder_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, config.classes)

    def embedded(self, tokens):
        b, h, w = tokens.shape
        if h != w:
            raise ValueError("Expected a square grid.")
        z = self.embedding(tokens.long()).flatten(1, 2)
        return z + positions(h, z.shape[-1], z.device, z.dtype)

    def latent(self, tokens):
        b, h, w = tokens.shape
        return self.pre_norm(self.embedded(tokens).transpose(1, 2).reshape(b, -1, h, w))

    def contribution(self, level, codes, size):
        z = self.books[level].lookup(codes).contiguous()
        z = self.coarse.decode(z, size) if level == 0 else resize(z, size)
        return .5 * z + .5 * self.phi[level](z)

    def quantize(self, z, update=False):
        full_size = z.shape[-1]
        cumulative = torch.zeros_like(z)
        codes, reconstructions = [], []
        for level, size in enumerate(self.config.sizes(full_size)):
            residual = z.detach() - cumulative.detach()
            vectors = self.coarse.encode(residual) if level == 0 else resize(residual, size)
            ids = self.books[level].nearest(vectors.detach(), update=update)
            q = self.books[level].lookup(ids).contiguous()
            if level == 0 and torch.is_grad_enabled():
                # ST trains the learned global downsampling operator too.
                # Preserve the code vector exactly in the forward pass. The
                # equivalent vectors + (q-vectors).detach() loses precision
                # when the learned downsampler produces large coordinates.
                q = (q.detach() + (vectors - vectors.detach())).contiguous()
                contribution = self.coarse.decode(q, full_size)
                contribution = .5 * contribution + .5 * self.phi[level](contribution)
            else:
                # Encoding frozen codes and sampling must share this exact
                # operator/layout. The training-only ST branch previously used
                # a different CUDA convolution path even under no_grad.
                contribution = self.contribution(level, ids, full_size)
            cumulative = cumulative + contribution
            codes.append(ids)
            reconstructions.append(cumulative)
        return codes, reconstructions

    @torch.no_grad()
    def encode(self, tokens):
        return self.quantize(self.latent(tokens), update=False)[0]

    def decode(self, z):
        b, d, h, w = z.shape
        x = z.flatten(2).transpose(1, 2)
        for block in self.decoder:
            x = block(x)
        return self.output(self.decoder_norm(x)).transpose(1, 2).reshape(b, -1, h, w)

    def loss(self, tokens, nested=True):
        z = self.latent(tokens)
        codes, reconstructions = self.quantize(z, update=self.training)
        cutoff = len(codes)
        if nested and torch.rand((), device=z.device) < self.config.nested_dropout:
            cutoff = int(torch.randint(1, cutoff, (), device=z.device))
        zq = reconstructions[cutoff - 1]
        # Paper equation 9 at the sampled nested cutoff, with EMA codebooks.
        vq = self.config.commitment * F.mse_loss(z, zq.detach()) + F.mse_loss(zq, z.detach())
        straight_through = zq.detach() + (z - z.detach())
        logits = self.decode(straight_through)
        ce = F.cross_entropy(logits, tokens.long())
        accuracy = (logits.argmax(1) == tokens).float().mean()
        exact = (logits.argmax(1) == tokens).flatten(1).all(1).float().mean()
        return ce + vq, {"ce": ce.detach(), "vq": vq.detach(), "accuracy": accuracy.detach(),
                        "exact_accuracy": exact.detach()}


class NextConceptGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.latent_dim, config.width)
        self.prefix_input = nn.Linear(config.latent_dim, config.width) if config.conditional else None
        self.level_embedding = nn.Embedding(len(config.scales), config.width)
        self.blocks = nn.ModuleList(AttentionBlock(config.width, config.heads, config.dropout)
                                    for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width)
        self.outputs = nn.ModuleList(nn.Linear(config.width, config.codebook_size)
                                     for _ in config.scales)

    def level_inputs(self, preceding, level, tokenizer):
        size = tokenizer.config.sizes(preceding.shape[-1])[level]
        x = self.input(resize(preceding, size).flatten(2).transpose(1, 2))
        x = x + positions(size, self.config.width, x.device, x.dtype)
        x = x + self.level_embedding.weight[level]
        return x

    def prefix_inputs(self, tokenizer, coloring):
        if self.config.conditional:
            if coloring is None:
                raise ValueError("Conditional NCM requires the original coloring.")
            with torch.no_grad():
                prefix = tokenizer.embedded(coloring)
            return self.prefix_input(prefix)
        return None

    def forward(self, preceding, level, tokenizer, coloring=None):
        size = tokenizer.config.sizes(preceding.shape[-1])[level]
        x = self.level_inputs(preceding, level, tokenizer)
        prefix = self.prefix_inputs(tokenizer, coloring)
        prefix_length = 0 if prefix is None else prefix.shape[1]
        if prefix is not None:
            x = torch.cat((prefix, x), 1)
        for block in self.blocks:
            x = block(x, prefix=prefix_length)
        logits = self.outputs[level](self.norm(x[:, prefix_length:]))
        return logits.transpose(1, 2).reshape(len(x), self.config.codebook_size, size, size)

    def forward_levels(self, preceding_levels, tokenizer, coloring=None):
        """All teacher-forced levels in one masked pass, as in NCM Algorithm 1."""
        levels = [self.level_inputs(z, k, tokenizer) for k, z in enumerate(preceding_levels)]
        prefix = self.prefix_inputs(tokenizer, coloring)
        prefix_length = 0 if prefix is None else prefix.shape[1]
        lengths = [x.shape[1] for x in levels]
        x = torch.cat(([prefix] if prefix is not None else []) + levels, dim=1)
        tags = torch.cat([torch.full((n,), k, device=x.device, dtype=torch.long)
                          for k, n in enumerate(lengths)])
        if prefix_length:
            tags = torch.cat((torch.full((prefix_length,), -1, device=x.device), tags))
        allowed = tags[:, None] == tags[None, :]
        if prefix_length:
            allowed[:, :prefix_length] = True
        for block in self.blocks:
            x = block(x, allowed_mask=allowed)
        chunks = self.norm(x[:, prefix_length:]).split(lengths, dim=1)
        sizes = tokenizer.config.sizes(preceding_levels[0].shape[-1])
        return [self.outputs[k](chunk).transpose(1, 2).reshape(len(x), self.config.codebook_size, s, s)
                for k, (chunk, s) in enumerate(zip(chunks, sizes))]


class GridNCM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = GridTokenizer(config)
        self.generator = NextConceptGenerator(config)

    def parameter_counts(self):
        tok = sum(p.numel() for p in self.tokenizer.parameters())
        gen = sum(p.numel() for p in self.generator.parameters())
        return {"tokenizer": tok, "generator": gen, "total": tok + gen}

    def generator_losses(self, target, coloring=None, corrupt=True):
        """Weighted next-level losses; sum before backward (packed graph is shared).

        The frozen tokenizer always runs in FP32, including under autocast, so
        teacher labels and inference reconstruction use the same code hierarchy.
        """
        self.tokenizer.eval()
        with torch.no_grad(), torch.autocast(target.device.type, enabled=False):
            latent = self.tokenizer.latent(target)
            codes, _ = self.tokenizer.quantize(latent, update=False)
            cumulative = torch.zeros_like(latent)
            preceding_levels = []
            for level, ids in enumerate(codes):
                preceding_levels.append(cumulative)
                conditioning_ids = ids
                if corrupt and self.config.corruption:
                    probability = self.config.corruption * level / max(1, len(codes) - 1)
                    mask = torch.rand(ids.shape, device=ids.device) < probability
                    random_ids = torch.randint(self.config.codebook_size, ids.shape, device=ids.device)
                    conditioning_ids = torch.where(mask, random_ids, ids)
                cumulative = cumulative + self.tokenizer.contribution(level, conditioning_ids, target.shape[-1])
        sizes = self.config.sizes(target.shape[-1])
        counts = [s * s for s in sizes]
        denominator = sum(n ** (1 - self.config.alpha) for n in counts)
        amp = getattr(self, "generator_precision", "float32") == "bfloat16"
        with torch.autocast(target.device.type, dtype=torch.bfloat16, enabled=amp):
            if getattr(self, "pack_levels", True):
                outputs = self.generator.forward_levels(preceding_levels, self.tokenizer, coloring)
            else:
                outputs = [self.generator(z, k, self.tokenizer, coloring)
                           for k, z in enumerate(preceding_levels)]
            losses = [F.cross_entropy(logits, ids) * count ** (1-self.config.alpha) / denominator
                      for logits, ids, count in zip(outputs, codes, counts)]
        yield from losses

    @torch.no_grad()
    def sample(self, count=None, *, size=None, coloring=None, temperature=1., seed=0,
               capture=False, progress=None):
        if temperature < 0 or not math.isfinite(temperature):
            raise ValueError("Temperature must be finite and nonnegative.")
        device = next(self.parameters()).device
        if coloring is not None:
            coloring = torch.as_tensor(coloring, dtype=torch.long, device=device)
            if coloring.ndim == 2:
                coloring = coloring[None]
            count, size = len(coloring), coloring.shape[-1]
        size = self.config.size if size is None else size
        if count is None or count < 1:
            raise ValueError("A positive sample count is required.")
        if self.config.conditional != (coloring is not None):
            raise ValueError("Coloring must be provided exactly for conditional models.")
        was_training = self.training
        self.eval()
        rng = torch.Generator(device=device).manual_seed(seed)
        cumulative = torch.zeros(count, self.config.latent_dim, size, size, device=device)
        frames = []
        try:
            for level in range(len(self.config.scales)):
                amp = getattr(self, "generator_precision", "float32") == "bfloat16"
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = self.generator(cumulative, level, self.tokenizer, coloring)
                if temperature == 0:
                    ids = logits.argmax(1)
                else:
                    probs = (logits.float() / temperature).softmax(1).permute(0, 2, 3, 1)
                    ids = torch.multinomial(probs.reshape(-1, self.config.codebook_size), 1,
                                            generator=rng).reshape(count, *probs.shape[1:3])
                cumulative = cumulative + self.tokenizer.contribution(level, ids, size)
                if capture or level == len(self.config.scales) - 1:
                    decoded = self.tokenizer.decode(cumulative)
                    if coloring is not None:
                        # Same color/EMPTY support as DFM; no queen-count or
                        # row/column constraints, search, or solution repair.
                        allowed = F.one_hot(coloring, self.config.classes).permute(0, 3, 1, 2).bool()
                        allowed[:, self.config.classes - 1] = True
                        decoded = decoded.masked_fill(~allowed, -torch.inf)
                    tokens = decoded.argmax(1).cpu()
                    if capture:
                        frames.append(tokens)
                if progress is not None:
                    progress(level + 1, len(self.config.scales))
            return (tokens, frames) if capture else tokens
        finally:
            self.train(was_training)
            self.tokenizer.eval()


def matched_config(config, target_parameters):
    """Select width/depth by total tokenizer+generator count, without allocating weights."""
    candidates = []
    with torch.device("meta"):
        for width in range(160, 321, 16):
            for layers in range(3, 9):
                candidate = replace(config, width=width, layers=layers)
                count = GridNCM(candidate).parameter_counts()["total"]
                candidates.append((abs(count - target_parameters), width, layers, candidate, count))
    _, _, _, chosen, count = min(candidates, key=lambda x: x[:3])
    return chosen, count
