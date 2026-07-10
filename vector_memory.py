import torch

class VectorRolloutBuffer:
    """
    Batched rollout buffer designed to gather parallel trajectories
    from multiple environments simultaneously and compute vectorized GAE.
    """
    def __init__(self, num_envs, rollout_steps, ppo_device):
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.device = ppo_device
        self.reset()

    def reset(self):
        # Initialize lists to store tensors of size (rollout_steps, num_envs, ...)
        self.states_latent = []
        self.states_scalar = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.terminals = []
        self.values = []
        self.action_masks = []

    def store(self, latents, scalars, actions, logprobs, rewards, terminals, values, masks):
        """Stores a single vectorized timestep transition."""
        self.states_latent.append(latents.clone().detach().to(self.device))
        self.states_scalar.append(scalars.clone().detach().to(self.device))
        self.actions.append(actions.clone().detach().to(self.device))
        self.logprobs.append(logprobs.clone().detach().to(self.device))
        self.rewards.append(torch.tensor(rewards, dtype=torch.float32, device=self.device))
        self.terminals.append(torch.tensor(terminals, dtype=torch.float32, device=self.device))
        self.values.append(values.clone().detach().squeeze(-1).to(self.device))
        self.action_masks.append(masks.clone().detach().to(self.device))

    def compute_gae(self, next_values, gamma, gae_lambda):
        """
        Vectorized implementation of Generalized Advantage Estimation (GAE).
        Processes advantages across all parallel environment tracks in parallel.
        """
        # Convert lists to stacked tensors of shape (rollout_steps, num_envs, ...)
        states_latent = torch.stack(self.states_latent)
        states_scalar = torch.stack(self.states_scalar)
        actions = torch.stack(self.actions)
        logprobs = torch.stack(self.logprobs)
        rewards = torch.stack(self.rewards)
        terminals = torch.stack(self.terminals)
        values = torch.stack(self.values)
        action_masks = torch.stack(self.action_masks)

        advantages = torch.zeros(self.rollout_steps, self.num_envs, device=self.device)
        last_gae_lam = 0.0

        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_non_terminal = 1.0 - terminals[step]
                next_val = next_values.squeeze(-1)
            else:
                next_non_terminal = 1.0 - terminals[step]
                next_val = values[step + 1]

            delta = rewards[step] + gamma * next_val * next_non_terminal - values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam

        returns = advantages + values

        # Flatten tensors to merge time and env dimensions (Batch Size = rollout_steps * num_envs)
        return (
            states_latent.view(-1, *states_latent.shape[2:]),
            states_scalar.view(-1, *states_scalar.shape[2:]),
            actions.view(-1),
            logprobs.view(-1),
            advantages.view(-1),
            returns.view(-1),
            action_masks.view(-1, *action_masks.shape[2:])
        )