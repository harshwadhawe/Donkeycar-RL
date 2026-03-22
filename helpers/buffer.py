"""
buffer.py — Episode Sequence Buffer for Dreamer
"""
import os
import numpy as np

class EpisodeBuffer:
    def __init__(self, max_steps, obs_shape, action_dim):
        self.max_steps = max_steps
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # Pre-allocate arrays
        self.obs = np.zeros((self.max_steps, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.max_steps, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.max_steps, dtype=np.float32)
        self.dones = np.zeros(self.max_steps, dtype=bool)
        
        self.ptr = 0
        self.size = 0
        self.total_steps = 0

    def add_step(self, obs, action, reward, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_steps
        self.size = min(self.size + 1, self.max_steps)
        self.total_steps += 1

    def sample(self, batch_size, chunk_size):
        """Samples contiguous sequences of shape (batch_size, chunk_size, ...)"""
        # Find valid starting indices (can't sample a chunk that wraps around or goes past self.size)
        valid_starts = np.arange(self.size - chunk_size)
        
        # Filter out starts where an episode terminates midway through the chunk
        # Dreamer can handle boundary resets, but for simplicity, we avoid crossing terminal states
        dones_view = sum(
            np.roll(self.dones[:self.size], -i) for i in range(chunk_size - 1)
        )
        valid_starts = valid_starts[dones_view[:len(valid_starts)] == 0]
        
        if len(valid_starts) < batch_size:
            # Fallback if buffer is too empty/fragmented
            idx = np.random.randint(0, self.size - chunk_size, size=batch_size)
        else:
            idx = np.random.choice(valid_starts, size=batch_size)

        obs_batch = np.stack([self.obs[i : i + chunk_size] for i in idx])
        act_batch = np.stack([self.actions[i : i + chunk_size] for i in idx])
        rew_batch = np.stack([self.rewards[i : i + chunk_size] for i in idx])
        done_batch = np.stack([self.dones[i : i + chunk_size] for i in idx])

        return obs_batch, act_batch, rew_batch, done_batch

    def save(self, filepath):
        """Saves buffer memory to a compressed NumPy archive."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not filepath.endswith('.npz'):
            filepath += '.npz'
            
        np.savez_compressed(
            filepath,
            obs=self.obs[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            dones=self.dones[:self.size]
        )

    def load(self, filepath):
        """Loads buffer memory from a compressed NumPy archive."""
        if not filepath.endswith('.npz'):
            filepath += '.npz'
            
        if os.path.exists(filepath):
            data = np.load(filepath)
            size = len(data['rewards'])
            self.obs[:size] = data['obs']
            self.actions[:size] = data['actions']
            self.rewards[:size] = data['rewards']
            self.dones[:size] = data['dones']
            
            self.size = size
            self.ptr = size % self.max_steps
            self.total_steps = size