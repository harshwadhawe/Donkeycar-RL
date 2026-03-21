"""
Dreamer v1 baseline for DonkeyCar.

Faithful port of github.com/ari-viitala/donkeycar + AaltoVision/donkeycar-dreamer
with bug fixes:
  - requires_gard typo fixed
  - hardcoded .to("cuda") replaced with device-agnostic code
  - print statements in select_action removed
  - SampleDist made device-aware

Architecture (Dreamer v1, Gaussian RSSM):
  - Belief (deterministic): GRU, 200 dims
  - State (stochastic): Gaussian, 30 dims
  - Encoder: 4-layer CNN -> 1024
  - Decoder: fc -> 4-layer transposed CNN
  - Reward: 3-layer MLP (ELU)
  - Value: 4-layer MLP (ELU), twin critics
  - Actor: 5-layer MLP (ELU), tanh-squashed Normal
  - Input: 1x40x40 grayscale, range [-0.5, 0.5]
"""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution


# ─── Config defaults (can be overridden via args) ────────────

DEFAULT_CONFIG = dict(
    belief_size=200,
    state_size=30,
    action_size=2,
    hidden_size=300,
    embedding_size=1024,
    observation_size=(1, 40, 40),
    planning_horizon=15,
    batch_size=50,
    chunk_size=50,
    discount=0.99,
    disclam=0.95,
    free_nats=3.0,
    world_lr=6e-4,
    actor_lr=8e-5,
    value_lr=8e-5,
    adam_epsilon=1e-7,
    grad_clip_norm=100.0,
    expl_amount=0.3,
    temp=0.003,
    reward_scale=10,
    pcont_scale=10,
    pcont=True,
    dense_act='elu',
    cnn_act='relu',
    bit_depth=8,
    experience_size=1_000_000,
    fix_speed=True,
    throttle_base=0.35,
    with_logprob=False,
    auto_temp=False,
    seed_episodes=5,
    gradient_steps=100,
    max_episode_steps=1000,
)


class DreamerV1Config:
    """Simple attribute-access config."""
    def __init__(self, device='cpu', **overrides):
        for k, v in DEFAULT_CONFIG.items():
            setattr(self, k, v)
        self.device = device
        for k, v in overrides.items():
            setattr(self, k, v)


# ─── Utility ────────────────────────────────────────────────

def bottle(f, x_tuple):
    """Apply f to (T*B, ...) shaped inputs, reshape back to (T, B, ...)."""
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]),
               zip(x_tuple, x_sizes)))
    y_size = y.size()
    return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])


def cal_returns(reward, value, bootstrap, pcont, lambda_):
    """Lambda-returns (Dreamer v1 eq. 5-6)."""
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    next_value = torch.cat((value[1:], bootstrap[None]), 0)
    inputs = reward + pcont * next_value * (1 - lambda_)
    outputs = []
    last = bootstrap
    for t in reversed(range(reward.shape[0])):
        last = inputs[t] + pcont[t] * lambda_ * last
        outputs.append(last)
    return torch.flip(torch.stack(outputs), [0])


# ─── Models ─────────────────────────────────────────────────

class TransitionModel(nn.Module):
    """Gaussian RSSM: belief (GRU) + stochastic state (Normal)."""

    def __init__(self, belief_size, state_size, action_size, hidden_size,
                 embedding_size, activation_function='relu', min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

    def forward(self, prev_state, actions, prev_belief, observations=None,
                nonterminals=None):
        T = actions.size(0) + 1
        beliefs = [torch.empty(0)] * T
        prior_states = [torch.empty(0)] * T
        prior_means = [torch.empty(0)] * T
        prior_std_devs = [torch.empty(0)] * T
        posterior_states = [torch.empty(0)] * T
        posterior_means = [torch.empty(0)] * T
        posterior_std_devs = [torch.empty(0)] * T

        beliefs[0] = prev_belief
        prior_states[0] = prev_state
        posterior_states[0] = prev_state

        for t in range(T - 1):
            _state = prior_states[t] if observations is None else posterior_states[t]
            if nonterminals is not None and t > 0:
                _state = _state * nonterminals[t - 1].unsqueeze(dim=-1)

            hidden = self.act_fn(
                self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1))
            )
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])

            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std = torch.chunk(
                self.fc_state_prior(hidden), 2, dim=1
            )
            prior_std_devs[t + 1] = F.softplus(_prior_std) + self.min_std_dev
            prior_states[t + 1] = (
                prior_means[t + 1] +
                prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            )

            if observations is not None:
                t_ = t - 1
                hidden = self.act_fn(self.fc_embed_belief_posterior(
                    torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)
                ))
                posterior_means[t + 1], _post_std = torch.chunk(
                    self.fc_state_posterior(hidden), 2, dim=1
                )
                posterior_std_devs[t + 1] = F.softplus(_post_std) + self.min_std_dev
                posterior_states[t + 1] = (
                    posterior_means[t + 1] +
                    posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
                )

        result = [
            torch.stack(beliefs[1:], dim=0),
            torch.stack(prior_states[1:], dim=0),
            torch.stack(prior_means[1:], dim=0),
            torch.stack(prior_std_devs[1:], dim=0),
        ]
        if observations is not None:
            result += [
                torch.stack(posterior_states[1:], dim=0),
                torch.stack(posterior_means[1:], dim=0),
                torch.stack(posterior_std_devs[1:], dim=0),
            ]
        return result


class VisualEncoder(nn.Module):
    """4-layer CNN: 1x40x40 -> 1024."""

    def __init__(self, embedding_size=1024, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.fc = (nn.Identity() if embedding_size == 1024
                   else nn.Linear(1024, embedding_size))

    def forward(self, observation):
        h = self.act_fn(self.conv1(observation))
        h = self.act_fn(self.conv2(h))
        h = self.act_fn(self.conv3(h))
        h = self.act_fn(self.conv4(h))
        h = h.view(-1, 1024)
        return self.fc(h)


class VisualDecoder(nn.Module):
    """4-layer transposed CNN: features -> 1x40x40."""

    def __init__(self, belief_size, state_size, embedding_size=1024,
                 activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 3, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 4, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 1, 6, stride=2)

    def forward(self, belief, state):
        h = self.fc1(torch.cat([belief, state], dim=1))
        h = h.view(-1, self.embedding_size, 1, 1)
        h = self.act_fn(self.conv1(h))
        h = self.act_fn(self.conv2(h))
        h = self.act_fn(self.conv3(h))
        return self.conv4(h)


class RewardModel(nn.Module):
    """3-layer MLP: (belief, state) -> scalar reward."""

    def __init__(self, belief_size, state_size, hidden_size,
                 activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        h = self.act_fn(self.fc1(x))
        h = self.act_fn(self.fc2(h))
        return self.fc3(h).squeeze(dim=-1)


class ValueModel(nn.Module):
    """4-layer MLP: (belief, state) -> scalar value."""

    def __init__(self, belief_size, state_size, hidden_size,
                 activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        h = self.act_fn(self.fc1(x))
        h = self.act_fn(self.fc2(h))
        h = self.act_fn(self.fc3(h))
        return self.fc4(h).squeeze(dim=1)


class PCONTModel(nn.Module):
    """Continuation predictor: (belief, state) -> P(continue)."""

    def __init__(self, belief_size, state_size, hidden_size,
                 activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, belief, state):
        x = torch.cat([belief, state], dim=1)
        h = self.act_fn(self.fc1(x))
        h = self.act_fn(self.fc2(h))
        h = self.act_fn(self.fc3(h))
        return torch.sigmoid(self.fc4(h).squeeze(dim=1))


class SampleDist:
    """Approximate mean/mode/entropy of a TransformedDistribution."""

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    def __getattr__(self, name):
        return getattr(self._dist, name)

    @property
    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        return torch.mean(dist.rsample(), 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = (torch.argmax(logprob, dim=0)
                   .reshape(1, batch_size, 1)
                   .expand(1, batch_size, feature_size))
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)


class ActorModel(nn.Module):
    """5-layer MLP with tanh-squashed Normal output."""

    def __init__(self, action_size, belief_size, state_size, hidden_size,
                 mean_scale=5, min_std=1e-4, init_std=5,
                 activation_function='elu',
                 fix_speed=False, throttle_base=0.3, device='cpu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fix_speed = fix_speed
        self.throttle_base = throttle_base
        self.mean_scale = mean_scale
        self.min_std = min_std
        self.init_std = init_std
        self._device = device

        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        out_dim = 2 * (action_size - 1) if fix_speed else 2 * action_size
        self.fc5 = nn.Linear(hidden_size, out_dim)

    def forward(self, belief, state, deterministic=False, with_logprob=False):
        raw_init_std = np.log(np.exp(self.init_std) - 1)
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=-1)))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        hidden = self.fc5(hidden)
        mean, std = torch.chunk(hidden, 2, dim=-1)

        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = F.softplus(std + raw_init_std) + self.min_std

        dist = Normal(mean, std)

        if self.fix_speed:
            transform = [AffineTransform(0., 2.), SigmoidTransform(),
                         AffineTransform(-1., 2.)]
        else:
            dev = belief.device
            transform = [
                AffineTransform(0., 2.), SigmoidTransform(),
                AffineTransform(-1., 2.),
                AffineTransform(
                    loc=torch.tensor([0.0, self.throttle_base], device=dev),
                    scale=torch.tensor([1.0, 0.2], device=dev),
                ),
            ]

        dist = TransformedDistribution(dist, transform)
        dist = SampleDist(dist)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        logp = None
        if with_logprob:
            logp = dist.log_prob(action).sum(dim=1)

        if self.fix_speed:
            throttle = torch.full_like(action[..., :1], self.throttle_base)
            action = torch.cat([action, throttle], dim=-1)

        return action, logp


# ─── Replay Buffer ──────────────────────────────────────────

class ExperienceReplay:
    """Flat circular replay buffer (numpy-backed)."""

    def __init__(self, size, observation_shape=(1, 40, 40), action_size=2,
                 device='cpu'):
        self.device = device
        self.size = size
        self.observations = np.empty((size, *observation_shape), dtype=np.float32)
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty(size, dtype=np.float32)
        self.nonterminals = np.empty(size, dtype=np.float32)
        self.idx = 0
        self.full = False
        self.steps = 0
        self.episodes = 0

    def append(self, observation, action, reward, done):
        if isinstance(observation, torch.Tensor):
            observation = observation.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        self.observations[self.idx] = observation
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = float(not done)
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1
        if done:
            self.episodes += 1

    def _sample_idx(self, L):
        while True:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            if self.idx not in idxs[1:]:
                return idxs

    def sample(self, n, L):
        idxs = np.asarray([self._sample_idx(L) for _ in range(n)])
        vec_idxs = idxs.transpose().reshape(-1)
        observations = torch.as_tensor(self.observations[vec_idxs])
        actions = torch.as_tensor(self.actions[vec_idxs])
        rewards = torch.as_tensor(self.rewards[vec_idxs])
        nonterminals = torch.as_tensor(self.nonterminals[vec_idxs])
        return [
            observations.reshape(L, n, *observations.shape[1:]).to(self.device),
            actions.reshape(L, n, -1).to(self.device),
            rewards.reshape(L, n).to(self.device),
            nonterminals.reshape(L, n).to(self.device),
        ]


# ─── Dreamer v1 Agent ───────────────────────────────────────

class DreamerV1:
    """Complete Dreamer v1 agent."""

    def __init__(self, device='cpu', **config_overrides):
        self.cfg = DreamerV1Config(device=device, **config_overrides)
        args = self.cfg
        self.device = device

        self.transition_model = TransitionModel(
            args.belief_size, args.state_size, args.action_size,
            args.hidden_size, args.embedding_size, args.dense_act,
        ).to(device)

        self.observation_model = VisualDecoder(
            args.belief_size, args.state_size, args.embedding_size,
            args.cnn_act,
        ).to(device)

        self.reward_model = RewardModel(
            args.belief_size, args.state_size, args.hidden_size,
            args.dense_act,
        ).to(device)

        self.encoder = VisualEncoder(args.embedding_size, args.cnn_act).to(device)

        self.actor_model = ActorModel(
            args.action_size, args.belief_size, args.state_size,
            args.hidden_size, activation_function=args.dense_act,
            fix_speed=args.fix_speed, throttle_base=args.throttle_base,
            device=device,
        ).to(device)

        self.value_model = ValueModel(
            args.belief_size, args.state_size, args.hidden_size,
            args.dense_act,
        ).to(device)

        self.value_model2 = ValueModel(
            args.belief_size, args.state_size, args.hidden_size,
            args.dense_act,
        ).to(device)

        self.pcont_model = PCONTModel(
            args.belief_size, args.state_size, args.hidden_size,
            args.dense_act,
        ).to(device) if args.pcont else None

        self.target_value_model = deepcopy(self.value_model)
        self.target_value_model2 = deepcopy(self.value_model2)
        for p in self.target_value_model.parameters():
            p.requires_grad = False
        for p in self.target_value_model2.parameters():
            p.requires_grad = False

        # World model params
        self.world_param = (
            list(self.transition_model.parameters()) +
            list(self.observation_model.parameters()) +
            list(self.reward_model.parameters()) +
            list(self.encoder.parameters())
        )
        if self.pcont_model is not None:
            self.world_param += list(self.pcont_model.parameters())

        # Optimizers
        self.world_optimizer = torch.optim.Adam(
            self.world_param, lr=args.world_lr, eps=args.adam_epsilon)
        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), lr=args.actor_lr, eps=args.adam_epsilon)
        self.value_optimizer = torch.optim.Adam(
            list(self.value_model.parameters()) +
            list(self.value_model2.parameters()),
            lr=args.value_lr, eps=args.adam_epsilon)

        self.free_nats = torch.full(
            (1,), args.free_nats, dtype=torch.float32, device=device)

        # Replay buffer
        self.D = ExperienceReplay(
            args.experience_size,
            observation_shape=args.observation_size,
            action_size=args.action_size,
            device=device,
        )

        # Belief state for online inference
        self.belief = torch.zeros(1, args.belief_size, device=device)
        self.posterior_state = torch.zeros(1, args.state_size, device=device)

    def reset_belief(self):
        self.belief = torch.zeros(1, self.cfg.belief_size, device=self.device)
        self.posterior_state = torch.zeros(1, self.cfg.state_size, device=self.device)

    def infer_state(self, observation, action):
        """Single-step belief update. observation: (1,1,40,40), action: (1,2)."""
        belief, _, _, _, posterior_state, _, _ = self.transition_model(
            self.posterior_state,
            action.unsqueeze(dim=0),
            self.belief,
            self.encoder(observation).unsqueeze(dim=0),
        )
        self.belief = belief.squeeze(dim=0)
        self.posterior_state = posterior_state.squeeze(dim=0)

    def select_action(self, observation, action, explore=True):
        """Select action given observation tensor and previous action tensor.

        Args:
            observation: (1, 1, 40, 40) tensor
            action: (1, 2) tensor (previous action)
            explore: if True, sample; if False, use mean

        Returns:
            numpy action array of shape (2,)
        """
        with torch.no_grad():
            observation = observation.to(self.device)
            action = action.to(self.device)
            self.infer_state(observation, action)

            act, _ = self.actor_model(
                self.belief, self.posterior_state,
                deterministic=not explore, with_logprob=False,
            )

            if explore and not self.cfg.with_logprob:
                act = Normal(act, self.cfg.expl_amount).rsample()
                act[:, 0].clamp_(-1.0, 1.0)
                if self.cfg.fix_speed:
                    act[:, 1] = self.cfg.throttle_base
                else:
                    act[:, 1].clamp_(0.0, 1.0)

        return act[0].cpu().numpy()

    def update_parameters(self, gradient_steps=None):
        """Train all components for gradient_steps iterations."""
        if gradient_steps is None:
            gradient_steps = self.cfg.gradient_steps
        args = self.cfg

        min_data = args.batch_size + args.chunk_size
        if self.D.steps < min_data:
            return {}

        totals = {k: 0.0 for k in [
            'obs_loss', 'reward_loss', 'kl_loss', 'pcont_loss',
            'actor_loss', 'value_loss',
        ]}

        for step in range(gradient_steps):
            observations, actions, rewards, nonterminals = self.D.sample(
                args.batch_size, args.chunk_size)

            init_belief = torch.zeros(
                args.batch_size, args.belief_size, device=self.device)
            init_state = torch.zeros(
                args.batch_size, args.state_size, device=self.device)

            # ── World Model ──────────────────────────────
            (beliefs, prior_states, prior_means, prior_std_devs,
             posterior_states, posterior_means, posterior_std_devs) = \
                self.transition_model(
                    init_state, actions, init_belief,
                    bottle(self.encoder, (observations,)),
                    nonterminals,
                )

            # Observation reconstruction
            obs_loss = F.mse_loss(
                bottle(self.observation_model, (beliefs, posterior_states)),
                observations,
                reduction='none',
            ).sum(dim=(2, 3, 4)).mean(dim=(0, 1))

            # Reward prediction
            reward_loss = args.reward_scale * F.mse_loss(
                bottle(self.reward_model, (beliefs, posterior_states)),
                rewards,
                reduction='none',
            ).mean(dim=(0, 1))

            # KL divergence with free nats
            from torch.distributions.independent import Independent
            from torch.distributions.kl import kl_divergence
            kl_loss = torch.max(
                kl_divergence(
                    Independent(Normal(posterior_means, posterior_std_devs), 1),
                    Independent(Normal(prior_means, prior_std_devs), 1),
                ),
                self.free_nats,
            ).mean(dim=(0, 1))

            # Continuation
            pcont_loss = torch.tensor(0.0, device=self.device)
            if self.pcont_model is not None:
                pcont_loss = args.pcont_scale * F.binary_cross_entropy(
                    bottle(self.pcont_model, (beliefs, posterior_states)),
                    nonterminals,
                )

            world_loss = obs_loss + reward_loss + kl_loss + pcont_loss
            self.world_optimizer.zero_grad()
            world_loss.backward()
            nn.utils.clip_grad_norm_(self.world_param, args.grad_clip_norm)
            self.world_optimizer.step()

            # ── Latent Imagination ────────────────────────
            for p in self.world_param:
                p.requires_grad = False
            for p in self.value_model.parameters():
                p.requires_grad = False
            for p in self.value_model2.parameters():
                p.requires_grad = False

            chunk_size, batch_size, _ = posterior_states.size()
            flatten_size = chunk_size * batch_size

            flat_beliefs = beliefs.detach().reshape(flatten_size, -1)
            flat_states = posterior_states.detach().reshape(flatten_size, -1)

            imag_beliefs = [flat_beliefs]
            imag_states = [flat_states]
            imag_ac_logps = []

            for _ in range(args.planning_horizon):
                imag_action, imag_logp = self.actor_model(
                    imag_beliefs[-1].detach(),
                    imag_states[-1].detach(),
                    deterministic=False,
                    with_logprob=args.with_logprob,
                )
                imag_action_t = imag_action.unsqueeze(dim=0)
                imag_belief, imag_state, _, _ = self.transition_model(
                    imag_states[-1], imag_action_t, imag_beliefs[-1])
                imag_beliefs.append(imag_belief.squeeze(dim=0))
                imag_states.append(imag_state.squeeze(dim=0))
                if args.with_logprob:
                    imag_ac_logps.append(imag_logp)

            imag_beliefs = torch.stack(imag_beliefs, dim=0)
            imag_states = torch.stack(imag_states, dim=0)
            imag_ac_logps_t = (torch.stack(imag_ac_logps, dim=0)
                               if args.with_logprob else None)

            # ── Actor Loss ────────────────────────────────
            imag_rewards = bottle(self.reward_model, (imag_beliefs, imag_states))
            imag_values = bottle(self.value_model, (imag_beliefs, imag_states))
            imag_values2 = bottle(self.value_model2, (imag_beliefs, imag_states))
            imag_values = torch.min(imag_values, imag_values2)

            with torch.no_grad():
                if self.pcont_model is not None:
                    pcont = bottle(self.pcont_model, (imag_beliefs, imag_states))
                else:
                    pcont = args.discount * torch.ones_like(imag_rewards)
            pcont = pcont.detach()

            if imag_ac_logps_t is not None:
                imag_values[1:] -= args.temp * imag_ac_logps_t

            returns = cal_returns(
                imag_rewards[:-1], imag_values[:-1], imag_values[-1],
                pcont[:-1], lambda_=args.disclam)

            discount = torch.cumprod(
                torch.cat([torch.ones_like(pcont[:1]), pcont[:-2]], 0), 0)
            discount = discount.detach()

            actor_loss = -torch.mean(discount * returns)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_model.parameters(), args.grad_clip_norm)
            self.actor_optimizer.step()

            # Unfreeze
            for p in self.world_param:
                p.requires_grad = True
            for p in self.value_model.parameters():
                p.requires_grad = True
            for p in self.value_model2.parameters():
                p.requires_grad = True

            # ── Critic Loss ───────────────────────────────
            imag_beliefs_d = imag_beliefs.detach()
            imag_states_d = imag_states.detach()

            with torch.no_grad():
                target_values = bottle(
                    self.target_value_model, (imag_beliefs_d, imag_states_d))
                target_values2 = bottle(
                    self.target_value_model2, (imag_beliefs_d, imag_states_d))
                target_values = torch.min(target_values, target_values2)
                target_rewards = bottle(
                    self.reward_model, (imag_beliefs_d, imag_states_d))

                if self.pcont_model is not None:
                    target_pcont = bottle(
                        self.pcont_model, (imag_beliefs_d, imag_states_d))
                else:
                    target_pcont = args.discount * torch.ones_like(target_rewards)

                if imag_ac_logps_t is not None:
                    target_values[1:] -= args.temp * imag_ac_logps_t.detach()

                target_returns = cal_returns(
                    target_rewards[:-1], target_values[:-1], target_values[-1],
                    target_pcont[:-1], lambda_=args.disclam)

            value_pred = bottle(
                self.value_model, (imag_beliefs_d, imag_states_d))[:-1]
            value_pred2 = bottle(
                self.value_model2, (imag_beliefs_d, imag_states_d))[:-1]

            value_loss = (
                F.mse_loss(value_pred, target_returns, reduction='none').mean() +
                F.mse_loss(value_pred2, target_returns, reduction='none').mean()
            )

            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(
                self.value_model.parameters(), args.grad_clip_norm)
            nn.utils.clip_grad_norm_(
                self.value_model2.parameters(), args.grad_clip_norm)
            self.value_optimizer.step()

            totals['obs_loss'] += obs_loss.item()
            totals['reward_loss'] += reward_loss.item()
            totals['kl_loss'] += kl_loss.item()
            totals['pcont_loss'] += (pcont_loss.item()
                                     if isinstance(pcont_loss, torch.Tensor)
                                     else pcont_loss)
            totals['actor_loss'] += actor_loss.item()
            totals['value_loss'] += value_loss.item()

        # Hard target update after all gradient steps
        with torch.no_grad():
            self.target_value_model.load_state_dict(
                self.value_model.state_dict())
            self.target_value_model2.load_state_dict(
                self.value_model2.state_dict())

        n = max(gradient_steps, 1)
        return {k: v / n for k, v in totals.items()}

    def save(self, path):
        state = {
            'transition_model': self.transition_model.state_dict(),
            'observation_model': self.observation_model.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'encoder': self.encoder.state_dict(),
            'actor_model': self.actor_model.state_dict(),
            'value_model': self.value_model.state_dict(),
            'value_model2': self.value_model2.state_dict(),
            'target_value_model': self.target_value_model.state_dict(),
            'target_value_model2': self.target_value_model2.state_dict(),
            'world_optimizer': self.world_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'version': 'dreamer_v1_baseline',
        }
        if self.pcont_model is not None:
            state['pcont_model'] = self.pcont_model.state_dict()
        torch.save(state, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.transition_model.load_state_dict(ckpt['transition_model'])
        self.observation_model.load_state_dict(ckpt['observation_model'])
        self.reward_model.load_state_dict(ckpt['reward_model'])
        self.encoder.load_state_dict(ckpt['encoder'])
        self.actor_model.load_state_dict(ckpt['actor_model'])
        self.value_model.load_state_dict(ckpt['value_model'])
        self.value_model2.load_state_dict(ckpt['value_model2'])
        if 'target_value_model' in ckpt:
            self.target_value_model.load_state_dict(ckpt['target_value_model'])
            self.target_value_model2.load_state_dict(ckpt['target_value_model2'])
        if 'world_optimizer' in ckpt:
            self.world_optimizer.load_state_dict(ckpt['world_optimizer'])
            self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
            self.value_optimizer.load_state_dict(ckpt['value_optimizer'])
        if self.pcont_model is not None and 'pcont_model' in ckpt:
            self.pcont_model.load_state_dict(ckpt['pcont_model'])
