"""
Dreamer v3: Model-Based Reinforcement Learning for DonkeyCar.

Implements Dreamer v3 (arXiv:2301.04104):
- Categorical RSSM (discrete latent state)
- Twohot discrete regression for reward and value heads
- Symlog predictions with zero-initialized output weights
- LayerNorm + SiLU throughout
- KL balancing with free nats (per-class)
- Percentile-based return normalization
- Slow EMA critic with value regularizer
- Dynamics-based actor gradients (continuous actions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, OneHotCategorical
import numpy as np

from . import config as cfg


# ─── Constants ───────────────────────────────────────────

NUM_BINS = 255
TWOHOT_LOW = -20.0
TWOHOT_HIGH = 20.0


# ─── Symlog Transform ─────────────────────────────────────

def symlog(x):
    """Symmetric log: sign(x) * ln(|x| + 1). Compresses large values."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# ─── Twohot Discrete Regression (Dreamer v3) ────────────

def twohot_encode(target, bins):
    """Encode scalar targets into soft two-hot distributions over bins.

    Args:
        target: (...) scalar values in symlog space
        bins:   (num_bins,) bin centers
    Returns:
        (..., num_bins) two-hot probability vectors
    """
    target = target.clamp(bins[0], bins[-1])
    below = torch.bucketize(target, bins) - 1
    below = below.clamp(0, len(bins) - 2)
    above = below + 1

    weight_above = (target - bins[below]) / (bins[above] - bins[below] + 1e-8)
    weight_below = 1.0 - weight_above

    dist = torch.zeros(*target.shape, len(bins), device=target.device)
    dist.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
    dist.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
    return dist


def twohot_decode(logits, bins):
    """Decode logits to scalar values via softmax-weighted bin sum.

    Args:
        logits: (..., num_bins)
        bins:   (num_bins,)
    Returns:
        (...) scalar values in symlog space
    """
    probs = F.softmax(logits, dim=-1)
    return (probs * bins).sum(dim=-1)


def twohot_loss(logits, target, bins):
    """Cross-entropy between predicted logits and two-hot encoded target.

    Args:
        logits: (..., num_bins)
        target: (...) scalar values in symlog space
        bins:   (num_bins,)
    Returns:
        (...) per-element loss
    """
    target_dist = twohot_encode(target, bins)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1)


# ─── MLP Block ────────────────────────────────────────────

class DreamerMLP(nn.Module):
    """MLP with LayerNorm + SiLU activations (Dreamer v3 style)."""

    def __init__(self, in_dim, out_dim, hidden_size, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = in_dim if i == 0 else hidden_size
            layers.append(nn.Linear(in_d, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_size, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Utility ──────────────────────────────────────────────

def bottle(fn, args):
    """Apply fn to (T*B, ...) shaped inputs, reshape back to (T, B, ...)."""
    t, b = args[0].shape[:2]
    reshaped = [a.reshape(t * b, *a.shape[2:]) for a in args]
    out = fn(*reshaped)
    if isinstance(out, tuple):
        return tuple(o.reshape(t, b, *o.shape[1:]) for o in out)
    return out.reshape(t, b, *out.shape[1:])


# ─── Visual Encoder ───────────────────────────────────────

class VisualEncoder(nn.Module):
    """CNN encoder: CxHxW -> embedding_size, with SiLU + projection."""

    def __init__(self, in_channels=1, embedding_size=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2), nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=1), nn.SiLU(),
        )
        # Compute flat size dynamically
        dummy = torch.zeros(1, in_channels, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        with torch.no_grad():
            flat_size = self.conv(dummy).flatten(1).shape[1]
        self.project = nn.Sequential(
            nn.Linear(flat_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.SiLU(),
        )
        self.embedding_size = embedding_size

    def forward(self, x):
        return self.project(self.conv(x).flatten(1))


# ─── Visual Decoder ───────────────────────────────────────

class VisualDecoder(nn.Module):
    """Transposed CNN decoder: features -> CxHxW image (any IMAGE_SIZE)."""

    def __init__(self, feature_size, out_channels=1):
        super().__init__()
        self.target_size = cfg.IMAGE_SIZE
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 256 * 4 * 4),
            nn.LayerNorm(256 * 4 * 4),
            nn.SiLU(),
        )
        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
        )

    def forward(self, features):
        h = self.fc(features)
        h = h.view(-1, 256, 4, 4)
        h = self.deconv(h)
        if h.shape[-1] != self.target_size:
            h = F.interpolate(h, size=(self.target_size, self.target_size),
                              mode='bilinear', align_corners=False)
        return h


# ─── Categorical RSSM ─────────────────────────────────────

class CategoricalRSSM(nn.Module):
    """
    Recurrent State Space Model with categorical latent state (Dreamer v3).

    Deterministic path: h_t = GRU(h_{t-1}, [s_{t-1}, a_{t-1}])
    Prior:     p(s_t | h_t)          = Cat(logits_prior)
    Posterior: q(s_t | h_t, o_t)     = Cat(logits_post)
    """

    def __init__(self, num_classes=16, num_categories=16, action_size=2,
                 belief_size=256, hidden_size=256, embedding_size=512,
                 unimix=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.num_categories = num_categories
        self.state_size = num_classes * num_categories
        self.belief_size = belief_size
        self.unimix = unimix

        self.fc_embed = nn.Sequential(
            nn.Linear(self.state_size + action_size, belief_size),
            nn.LayerNorm(belief_size),
            nn.SiLU(),
        )
        self.rnn = nn.GRUCell(belief_size, belief_size)

        self.fc_prior = nn.Sequential(
            nn.Linear(belief_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_classes * num_categories),
        )

        self.fc_posterior = nn.Sequential(
            nn.Linear(belief_size + embedding_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_classes * num_categories),
        )

    def forward(self, prev_state, actions, prev_belief, observations=None):
        T = actions.shape[0]
        beliefs = []
        prior_logits_all, prior_states_all = [], []
        posterior_logits_all, posterior_states_all = [], []

        belief = prev_belief
        state = prev_state

        for t in range(T):
            x = self.fc_embed(torch.cat([state, actions[t]], dim=-1))
            belief = self.rnn(x, belief)

            prior_logits = self.fc_prior(belief).view(
                -1, self.num_classes, self.num_categories
            )
            prior_state_oh = self._sample(prior_logits)
            prior_state = prior_state_oh.flatten(start_dim=1)

            if observations is not None:
                post_input = torch.cat([belief, observations[t]], dim=-1)
                post_logits = self.fc_posterior(post_input).view(
                    -1, self.num_classes, self.num_categories
                )
                post_state_oh = self._sample(post_logits)
                state = post_state_oh.flatten(start_dim=1)
                posterior_logits_all.append(post_logits)
                posterior_states_all.append(state)
            else:
                state = prior_state

            beliefs.append(belief)
            prior_logits_all.append(prior_logits)
            prior_states_all.append(state if observations is None else prior_state)

        result = {
            'beliefs': torch.stack(beliefs),
            'prior_logits': torch.stack(prior_logits_all),
            'prior_states': torch.stack(prior_states_all),
        }
        if observations is not None:
            result['posterior_logits'] = torch.stack(posterior_logits_all)
            result['posterior_states'] = torch.stack(posterior_states_all)
        return result

    def _sample(self, logits):
        """Sample from categorical with unimix and straight-through gradient."""
        if self.unimix > 0:
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / self.num_categories
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = torch.log(probs + 1e-8)

        probs = F.softmax(logits, dim=-1)
        dist = OneHotCategorical(probs=probs)
        sample = dist.sample()
        return sample + probs - probs.detach()

    def imagine_step(self, state, action, belief):
        """Single imagination step (no observation). Returns new belief, state."""
        x = self.fc_embed(torch.cat([state, action], dim=-1))
        belief = self.rnn(x, belief)
        prior_logits = self.fc_prior(belief).view(
            -1, self.num_classes, self.num_categories
        )
        state = self._sample(prior_logits).flatten(start_dim=1)
        return belief, state


# ─── Reward, Continue, and Value Models ────────────────────

class RewardModel(nn.Module):
    """Predicts reward as twohot distribution over symlog bins."""

    def __init__(self, feature_size, hidden_size=256, num_layers=2):
        super().__init__()
        self.mlp = DreamerMLP(feature_size, NUM_BINS, hidden_size, num_layers)
        # Zero-init output weights (Dreamer v3 paper §3.1)
        self.mlp.net[-1].weight.data.zero_()
        self.mlp.net[-1].bias.data.zero_()

    def forward(self, features):
        return self.mlp(features)  # (*, NUM_BINS) logits


class ContinueModel(nn.Module):
    """Predicts P(episode continues) from features. Sigmoid output."""

    def __init__(self, feature_size, hidden_size=256, num_layers=2):
        super().__init__()
        self.mlp = DreamerMLP(feature_size, 1, hidden_size, num_layers)

    def forward(self, features):
        return torch.sigmoid(self.mlp(features))


class ValueModel(nn.Module):
    """Predicts value as twohot distribution over symlog bins."""

    def __init__(self, feature_size, hidden_size=256, num_layers=3):
        super().__init__()
        self.mlp = DreamerMLP(feature_size, NUM_BINS, hidden_size, num_layers)
        # Zero-init output weights (Dreamer v3 paper §3.1)
        self.mlp.net[-1].weight.data.zero_()
        self.mlp.net[-1].bias.data.zero_()

    def forward(self, features):
        return self.mlp(features)  # (*, NUM_BINS) logits


# ─── Actor Model ──────────────────────────────────────────

class ActorModel(nn.Module):
    """
    Policy network: (belief, state) -> Normal distribution over actions.
    Uses tanh squashing. LayerNorm + SiLU hidden layers.
    """

    def __init__(self, feature_size, action_size=2, hidden_size=256,
                 num_layers=3, min_std=0.1, init_std=1.0,
                 fix_speed=False, throttle_base=0.3):
        super().__init__()
        self.action_size = action_size
        self.min_std = min_std
        self.raw_init_std = np.log(np.exp(init_std) - 1)  # inverse softplus
        self.fix_speed = fix_speed
        self.throttle_base = throttle_base

        self._learned_dims = 1 if fix_speed else action_size
        self.mlp = DreamerMLP(feature_size, 2 * self._learned_dims, hidden_size, num_layers)

    def forward(self, features):
        out = self.mlp(features)
        mean, std_raw = out.chunk(2, dim=-1)
        mean = 5.0 * torch.tanh(mean / 5.0)
        std = F.softplus(std_raw + self.raw_init_std) + self.min_std
        return mean, std

    def get_dist(self, features):
        mean, std = self.forward(features)
        return Normal(mean, std)

    def sample_action(self, features, explore=False, **kwargs):
        dist = self.get_dist(features)
        if explore:
            action = dist.rsample()
        else:
            action = dist.mean
        action = torch.tanh(action)

        if self.fix_speed:
            throttle = torch.full_like(action[..., :1], self.throttle_base)
            action = torch.cat([action, throttle], dim=-1)
        return action

    def log_prob(self, features, action):
        if self.fix_speed:
            action = action[..., :1]
        dist = self.get_dist(features)
        raw_action = torch.atanh(action.clamp(-0.999, 0.999))
        log_p = dist.log_prob(raw_action)
        log_p = log_p - torch.log(1 - action.pow(2) + 1e-6)
        return log_p.sum(dim=-1, keepdim=True)

    def entropy(self, features):
        dist = self.get_dist(features)
        return dist.entropy().sum(dim=-1, keepdim=True)


# ─── Lambda Returns ───────────────────────────────────────

def compute_lambda_returns(rewards, values, continues, gamma, disclam):
    """
    Compute lambda-returns V_lambda.

    Args:
        rewards:   (H, B, 1)
        values:    (H+1, B, 1)
        continues: (H, B, 1)
        gamma, disclam: scalars
    Returns:
        returns:   (H, B, 1)
    """
    H = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    last = values[-1]

    for t in reversed(range(H)):
        returns[t] = rewards[t] + gamma * continues[t] * (
            (1 - disclam) * values[t + 1] + disclam * last
        )
        last = returns[t]
    return returns


# ─── Dreamer v3 Agent ─────────────────────────────────────

class Dreamer:
    """
    Complete Dreamer v3 agent with twohot regression, slow value
    regularizer, and dynamics-based actor gradients.
    """

    def __init__(self, device='cpu'):
        self.device = device
        action_size = 2
        channels = 1 if not cfg.RGB else 3
        bs = cfg.DREAMER_BELIEF_SIZE
        nc = cfg.DREAMER_NUM_CLASSES
        ncat = cfg.DREAMER_NUM_CATEGORIES
        ss = nc * ncat
        hs = cfg.DREAMER_HIDDEN_SIZE
        es = cfg.DREAMER_EMBEDDING_SIZE
        feature_size = bs + ss

        # Twohot bins (registered once, moved to device)
        self._twohot_bins = torch.linspace(
            TWOHOT_LOW, TWOHOT_HIGH, NUM_BINS, device=device
        )

        # World model
        self.encoder = VisualEncoder(channels, es).to(device)
        self.decoder = VisualDecoder(feature_size, channels).to(device)
        self.rssm = CategoricalRSSM(
            nc, ncat, action_size, bs, hs, es, cfg.DREAMER_UNIMIX
        ).to(device)
        self.reward_model = RewardModel(feature_size, hs).to(device)
        self.continue_model = ContinueModel(feature_size, hs).to(device) \
            if cfg.DREAMER_PCONT else None

        # Actor-critic
        self.actor = ActorModel(
            feature_size, action_size, hs,
            min_std=cfg.DREAMER_ACTOR_MIN_STD,
            init_std=cfg.DREAMER_ACTOR_INIT_STD,
            fix_speed=cfg.DREAMER_FIX_SPEED,
            throttle_base=cfg.DREAMER_THROTTLE_BASE,
        ).to(device)
        self.value_model = ValueModel(feature_size, hs).to(device)
        self.value_target = ValueModel(feature_size, hs).to(device)
        self.value_target.load_state_dict(self.value_model.state_dict())

        # Optimizers
        world_params = self._world_params()
        self.world_optimizer = torch.optim.Adam(
            world_params, lr=cfg.DREAMER_WORLD_LR, eps=cfg.DREAMER_ADAM_EPS
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.DREAMER_ACTOR_LR, eps=cfg.DREAMER_ADAM_EPS
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_model.parameters(), lr=cfg.DREAMER_VALUE_LR, eps=cfg.DREAMER_ADAM_EPS
        )

        # Belief state for real-time inference
        self.belief = torch.zeros(1, bs, device=device)
        self.state = torch.zeros(1, ss, device=device)

        self._gradient_step_count = 0

    def reset_belief(self):
        """Reset belief state at episode start."""
        bs = cfg.DREAMER_BELIEF_SIZE
        ss = cfg.DREAMER_NUM_CLASSES * cfg.DREAMER_NUM_CATEGORIES
        self.belief = torch.zeros(1, bs, device=self.device)
        self.state = torch.zeros(1, ss, device=self.device)

    def infer_state(self, observation, action):
        """Single-step belief update for real-time inference."""
        with torch.no_grad():
            observation = observation.to(self.device)
            action = action.to(self.device)
            embedding = self.encoder(observation)
            action_seq = action.unsqueeze(0)
            embedding_seq = embedding.unsqueeze(0)

            result = self.rssm(
                self.state, action_seq, self.belief, embedding_seq
            )
            self.belief = result['beliefs'][0]
            self.state = result['posterior_states'][0]

    def select_action(self, observation, action, explore=False):
        """Select action given observation and previous action."""
        self.infer_state(observation, action)
        with torch.no_grad():
            features = torch.cat([self.belief, self.state], dim=-1)
            action = self.actor.sample_action(features, explore=explore)
        return action.cpu().numpy().flatten()

    def update(self, buffer, gradient_steps=None):
        """Train Dreamer v3 on replay buffer data."""
        if gradient_steps is None:
            gradient_steps = cfg.DREAMER_GRADIENT_STEPS

        if buffer.total_steps < cfg.DREAMER_CHUNK_SIZE * 2:
            return {}

        bins = self._twohot_bins

        totals = {k: 0.0 for k in [
            'world_loss', 'obs_loss', 'reward_loss', 'kl_loss',
            'actor_loss', 'value_loss'
        ]}

        for step in range(gradient_steps):
            data = buffer.sample_chunks(cfg.DREAMER_BATCH_SIZE, cfg.DREAMER_CHUNK_SIZE)
            if data is None:
                break
            obs, actions, rewards, dones = [d.to(self.device) for d in data]
            T, B = obs.shape[:2]

            # ── World Model ──────────────────────────────
            embeddings = bottle(self.encoder, [obs])

            bs = cfg.DREAMER_BELIEF_SIZE
            ss = cfg.DREAMER_NUM_CLASSES * cfg.DREAMER_NUM_CATEGORIES
            init_belief = torch.zeros(B, bs, device=self.device)
            init_state = torch.zeros(B, ss, device=self.device)

            result = self.rssm(init_state, actions, init_belief, embeddings)
            beliefs = result['beliefs']
            posterior_states = result['posterior_states']
            prior_logits = result['prior_logits']
            posterior_logits = result['posterior_logits']

            features = torch.cat([beliefs, posterior_states], dim=-1)

            # Observation reconstruction loss
            recon = bottle(self.decoder, [features])
            obs_loss = F.mse_loss(recon, obs, reduction='none').sum(dim=(2, 3, 4)).mean()

            # Reward loss — twohot cross-entropy in symlog space
            pred_reward_logits = bottle(self.reward_model, [features])  # (T, B, NUM_BINS)
            reward_targets = symlog(rewards)  # (T, B)
            reward_loss = twohot_loss(
                pred_reward_logits.reshape(-1, NUM_BINS),
                reward_targets.reshape(-1),
                bins
            ).mean()

            # KL balancing (Dreamer v3) — free nats applied per class
            nc = cfg.DREAMER_NUM_CLASSES

            post_dist = OneHotCategorical(logits=posterior_logits)
            prior_dist_sg = OneHotCategorical(logits=prior_logits.detach())
            kl_rep = torch.distributions.kl_divergence(
                post_dist, prior_dist_sg
            )  # (T, B, num_classes)
            # Free nats per class, then sum
            kl_rep = torch.clamp(kl_rep, min=cfg.DREAMER_KL_FREE_NATS / nc).sum(dim=-1)

            post_dist_sg = OneHotCategorical(logits=posterior_logits.detach())
            prior_dist = OneHotCategorical(logits=prior_logits)
            kl_dyn = torch.distributions.kl_divergence(
                post_dist_sg, prior_dist
            )  # (T, B, num_classes)
            kl_dyn = torch.clamp(kl_dyn, min=cfg.DREAMER_KL_FREE_NATS / nc).sum(dim=-1)

            kl_loss = (
                cfg.DREAMER_KL_REP_WEIGHT * kl_rep.mean() +
                cfg.DREAMER_KL_DYN_WEIGHT * kl_dyn.mean()
            )

            # Continue model loss
            cont_loss = torch.tensor(0.0, device=self.device)
            if self.continue_model is not None:
                pred_cont = bottle(self.continue_model, [features])
                cont_target = (1 - dones).unsqueeze(-1)
                cont_loss = F.binary_cross_entropy(
                    pred_cont, cont_target, reduction='mean'
                )

            world_loss = obs_loss + reward_loss + kl_loss + cont_loss

            self.world_optimizer.zero_grad()
            world_loss.backward()
            nn.utils.clip_grad_norm_(self._world_params(), cfg.DREAMER_GRAD_CLIP)
            self.world_optimizer.step()

            # ── Actor (Latent Imagination) ───────────────
            for p in self._world_params():
                p.requires_grad_(False)
            for p in self.value_model.parameters():
                p.requires_grad_(False)

            flat_belief = beliefs.detach().reshape(-1, bs)
            flat_state = posterior_states.detach().reshape(-1, ss)

            imag_beliefs = [flat_belief]
            imag_states = [flat_state]

            for _ in range(cfg.DREAMER_PLANNING_HORIZON):
                imag_feat = torch.cat(
                    [imag_beliefs[-1], imag_states[-1]], dim=-1
                )
                imag_action = self.actor.sample_action(imag_feat, explore=False)
                new_belief, new_state = self.rssm.imagine_step(
                    imag_states[-1], imag_action, imag_beliefs[-1]
                )
                imag_beliefs.append(new_belief)
                imag_states.append(new_state)

            imag_beliefs = torch.stack(imag_beliefs)     # (H+1, T*B, bs)
            imag_states = torch.stack(imag_states)       # (H+1, T*B, ss)
            imag_features = torch.cat([imag_beliefs, imag_states], dim=-1)

            # Predict rewards and values via twohot decoding
            imag_reward_logits = bottle(
                self.reward_model, [imag_features[:-1]]
            )  # (H, T*B, NUM_BINS)
            imag_rewards = symexp(
                twohot_decode(imag_reward_logits, bins)
            ).unsqueeze(-1)  # (H, T*B, 1)

            imag_value_logits = bottle(
                self.value_target, [imag_features]
            )  # (H+1, T*B, NUM_BINS)
            imag_values = symexp(
                twohot_decode(imag_value_logits, bins)
            ).unsqueeze(-1)  # (H+1, T*B, 1)

            if self.continue_model is not None:
                imag_cont = bottle(
                    self.continue_model, [imag_features[:-1]]
                )
            else:
                imag_cont = torch.ones_like(imag_rewards)

            # Lambda returns
            returns = compute_lambda_returns(
                imag_rewards, imag_values, imag_cont,
                cfg.DREAMER_GAMMA, cfg.DREAMER_DISCLAM
            )  # (H, T*B, 1)

            # Compute advantages (return - value baseline)
            # baseline is detached; returns keeps gradients through actor
            with torch.no_grad():
                baseline_logits = bottle(
                    self.value_model, [imag_features[:-1]]
                )
                baseline = symexp(
                    twohot_decode(baseline_logits, bins)
                ).unsqueeze(-1)  # (H, T*B, 1)

            advantages = returns - baseline  # returns has grads, baseline detached

            # Percentile normalization (compute scale from detached values)
            with torch.no_grad():
                flat_adv = advantages.flatten()
                low = torch.quantile(flat_adv, cfg.DREAMER_RETURN_NORM_LOW / 100.0)
                high = torch.quantile(flat_adv, cfg.DREAMER_RETURN_NORM_HIGH / 100.0)
                scale = torch.clamp(high - low, min=1.0)
            # Apply scale but keep gradient through returns
            normalized_adv = (advantages - low.detach()) / scale.detach()

            # Actor loss: dynamics-based gradient flows through returns → actor
            actor_entropy = bottle(
                self.actor.entropy, [imag_features[:-1]]
            )  # (H, T*B, 1)
            actor_loss = -(
                normalized_adv + cfg.DREAMER_ACTOR_ENTROPY * actor_entropy
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.DREAMER_GRAD_CLIP)
            self.actor_optimizer.step()

            # Unfreeze
            for p in self._world_params():
                p.requires_grad_(True)
            for p in self.value_model.parameters():
                p.requires_grad_(True)

            # ── Value Model ──────────────────────────────
            # Twohot cross-entropy loss
            value_logits = bottle(
                self.value_model,
                [imag_features[:-1].detach()]
            )  # (H, T*B, NUM_BINS)
            value_targets = symlog(returns.squeeze(-1)).detach()  # (H, T*B)
            value_loss = twohot_loss(
                value_logits.reshape(-1, NUM_BINS),
                value_targets.reshape(-1),
                bins
            ).mean()

            # Slow value regularizer: cross-entropy toward EMA target
            with torch.no_grad():
                slow_logits = bottle(
                    self.value_target,
                    [imag_features[:-1].detach()]
                )
                slow_probs = F.softmax(slow_logits, dim=-1)
            value_reg = -(
                slow_probs.reshape(-1, NUM_BINS) *
                F.log_softmax(value_logits.reshape(-1, NUM_BINS), dim=-1)
            ).sum(dim=-1).mean()
            value_loss += value_reg

            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                self.value_model.parameters(), cfg.DREAMER_GRAD_CLIP
            )
            self.value_optimizer.step()

            # Slow EMA target update
            tau = cfg.DREAMER_SLOW_TARGET_FRACTION
            for p, tp in zip(self.value_model.parameters(),
                             self.value_target.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

            self._gradient_step_count += 1

            totals['world_loss'] += world_loss.item()
            totals['obs_loss'] += obs_loss.item()
            totals['reward_loss'] += reward_loss.item()
            totals['kl_loss'] += kl_loss.item()
            totals['actor_loss'] += actor_loss.item()
            totals['value_loss'] += value_loss.item()

        n = max(gradient_steps, 1)
        return {k: v / n for k, v in totals.items()}

    def _world_params(self):
        params = (
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.reward_model.parameters())
        )
        if self.continue_model:
            params += list(self.continue_model.parameters())
        return params

    def save(self, path):
        state = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'rssm': self.rssm.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'actor': self.actor.state_dict(),
            'value_model': self.value_model.state_dict(),
            'value_target': self.value_target.state_dict(),
            'version': 'dreamer_v3_twohot',
        }
        if self.continue_model:
            state['continue_model'] = self.continue_model.state_dict()
        torch.save(state, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.reward_model.load_state_dict(checkpoint['reward_model'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.value_model.load_state_dict(checkpoint['value_model'])
        self.value_target.load_state_dict(checkpoint['value_target'])
        if self.continue_model and 'continue_model' in checkpoint:
            self.continue_model.load_state_dict(checkpoint['continue_model'])
