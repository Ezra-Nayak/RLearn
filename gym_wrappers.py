import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class FrameStackWrapper(gym.ObservationWrapper):
    """
    Gymnasium Observation Wrapper that stacks the last 4 grayscale images.
    Returns:
        Dict containing:
            "image": Stacked float32 tensor of shape (4, 160, 160)
            "scalars": Proprioception features [norm_x, eagle_threat]
    """

    def __init__(self, env, stack_size=4):
        super(FrameStackWrapper, self).__init__(env)
        self.stack_size = stack_size
        self.frame_buffer = deque(maxlen=stack_size)

        # Modify the observation space to account for stacked channels
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0.0, high=1.0, shape=(stack_size, 160, 160), dtype=np.float32),
            "scalars": env.observation_space["scalars"]
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frame_buffer.clear()

        # Pre-fill the stack with the initial frame
        initial_frame = obs["image"]
        for _ in range(self.stack_size):
            self.frame_buffer.append(initial_frame)

        obs["image"] = np.array(self.frame_buffer, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Append the new frame and construct the stack
        self.frame_buffer.append(obs["image"])
        obs["image"] = np.array(self.frame_buffer, dtype=np.float32)

        return obs, reward, terminated, truncated, info