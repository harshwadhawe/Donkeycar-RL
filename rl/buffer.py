"""
Replay buffers for RL agents.

- ReplayBuffer: flat transition buffer for SAC
- EpisodeBuffer: episode-based sequential buffer for Dreamer (RSSM needs sequences)
"""

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Flat transition buffer for SAC+VAE.
    Stores (image, cmd_history, action, reward, next_image, next_cmd, done).
    """

    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)

    def push(self, image, cmd_history, action, reward, next_image, next_cmd, done):
        self.buffer.append((
            image.cpu(), cmd_history.cpu(), action.cpu(),
            torch.tensor([reward], dtype=torch.float32),
            next_image.cpu(), next_cmd.cpu(),
            torch.tensor([1.0 - float(done)], dtype=torch.float32)
        ))

    def sample(self, batch_size):
        batch = random.choices(self.buffer, k=batch_size)
        images, cmds, actions, rewards, next_images, next_cmds, not_dones = zip(*batch)
        return (
            torch.stack(images), torch.stack(cmds), torch.stack(actions),
            torch.stack(rewards), torch.stack(next_images), torch.stack(next_cmds),
            torch.stack(not_dones)
        )

    def __len__(self):
        return len(self.buffer)


class EpisodeBuffer:
    """
    Episode-based replay buffer for Dreamer.

    Stores complete episodes as dicts of {observations, actions, rewards, dones}.
    Samples random chunks of length chunk_size for RSSM training.
    """

    def __init__(self, max_steps=1_000_000, obs_shape=(1, 40, 40), action_dim=2):
        self.max_steps = max_steps
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        # Storage as flat arrays (ring buffer over total steps)
        self.observations = np.zeros((max_steps, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((max_steps, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_steps, dtype=np.float32)
        self.dones = np.zeros(max_steps, dtype=np.float32)

        self.idx = 0        # current write position
        self.full = False    # whether buffer has wrapped
        # Each entry is (start_idx, end_idx) for completed episodes
        self._episodes = []
        self._current_start = None

    @property
    def total_steps(self):
        return self.max_steps if self.full else self.idx

    @property
    def num_episodes(self):
        return len(self._episodes)

    @property
    def episode_starts(self):
        """For backward compatibility (logging)."""
        return [s for s, e in self._episodes]

    def add_step(self, obs, action, reward, done):
        """
        Add a single step. obs is a numpy array of shape obs_shape.
        Call with done=True to mark episode end.
        """
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = float(done)

        if self._current_start is None:
            self._current_start = self.idx

        self.idx = (self.idx + 1) % self.max_steps
        if self.idx == 0:
            self.full = True

        # Prune episodes whose data has been overwritten by the ring buffer
        if self.full:
            self._episodes = [
                (s, e) for s, e in self._episodes
                if not self._is_overwritten(s, e)
            ]

        if done:
            # Record completed episode with its end index (inclusive)
            end_idx = (self.idx - 1) % self.max_steps
            self._episodes.append((self._current_start, end_idx))
            self._current_start = None

    def _is_overwritten(self, start, end):
        """
        Check if an episode [start..end] has been partially overwritten.
        The write pointer (self.idx) is the next position to be written.
        After wraparound, valid data is [self.idx .. self.idx-1] mod max_steps.
        An episode is overwritten if self.idx has advanced past its start.
        Since episodes are appended chronologically, we check if start is
        in the overwritten zone by comparing ring-buffer distances.
        """
        # Distance from write pointer to episode start (going forward in the ring)
        # If this distance exceeds the buffer capacity minus 1, the start was overwritten.
        # Equivalently: the start is valid if it's "ahead" of the write pointer.
        dist = (start - self.idx) % self.max_steps
        # dist == 0 means start is exactly at write pointer (about to be overwritten)
        # A valid start should have dist > 0 (it's in the valid region ahead of write ptr)
        return dist == 0

    def sample_chunks(self, batch_size, chunk_size):
        """
        Sample batch_size random chunks of length chunk_size.

        Returns:
            observations: (chunk_size, batch_size, *obs_shape)
            actions:      (chunk_size, batch_size, action_dim)
            rewards:      (chunk_size, batch_size)
            dones:        (chunk_size, batch_size)
        """
        # Find valid start positions from completed episodes
        valid_starts = []
        for ep_start, ep_end in self._episodes:
            # Compute episode length accounting for ring buffer wrap
            ep_len = (ep_end - ep_start) % self.max_steps + 1

            # Valid starts: anywhere in episode where chunk_size fits
            if ep_len >= chunk_size:
                for offset in range(ep_len - chunk_size + 1):
                    valid_starts.append((ep_start + offset) % self.max_steps)

        if len(valid_starts) == 0:
            return None

        # Sample batch_size starts
        chosen = random.choices(valid_starts, k=batch_size)

        obs_chunks = np.zeros((chunk_size, batch_size, *self.obs_shape), dtype=np.float32)
        act_chunks = np.zeros((chunk_size, batch_size, self.action_dim), dtype=np.float32)
        rew_chunks = np.zeros((chunk_size, batch_size), dtype=np.float32)
        done_chunks = np.zeros((chunk_size, batch_size), dtype=np.float32)

        for b, start in enumerate(chosen):
            for t in range(chunk_size):
                idx = (start + t) % self.max_steps
                obs_chunks[t, b] = self.observations[idx]
                act_chunks[t, b] = self.actions[idx]
                rew_chunks[t, b] = self.rewards[idx]
                done_chunks[t, b] = self.dones[idx]

        return (
            torch.from_numpy(obs_chunks),
            torch.from_numpy(act_chunks),
            torch.from_numpy(rew_chunks),
            torch.from_numpy(done_chunks),
        )
