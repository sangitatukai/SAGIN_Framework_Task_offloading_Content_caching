# ppo_gru_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GRUPPOAgent(nn.Module):
    def __init__(self, obs_dim, num_contents, hidden_dim=64, gamma=0.99, clip_eps=0.2, lr=1e-3):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_contents = num_contents
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.lr = lr

        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.cache_fc = nn.Linear(hidden_dim, num_contents)
        self.offload_fc = nn.Linear(hidden_dim, 1)
        self.value_fc = nn.Linear(hidden_dim, 1)

        self.hidden_state = None
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.buffer = []  # (state, action, logprob, reward, value, done)

    def forward(self, obs_seq):
        out, self.hidden_state = self.gru(obs_seq, self.hidden_state)
        last_out = out[:, -1, :]
        cache_logits = self.cache_fc(last_out)
        offload_logit = self.offload_fc(last_out)
        value = self.value_fc(last_out)

        cache_probs = torch.softmax(cache_logits, dim=-1)
        offload_prob = torch.sigmoid(offload_logit)
        return cache_probs, offload_prob, value

    def act(self, obs_np):
        obs = torch.from_numpy(obs_np).float().unsqueeze(0).unsqueeze(0)  # [1, 1, obs_dim]
        self.hidden_state = None  # Reset hidden state for independent decisions
        with torch.no_grad():
            cache_probs, offload_prob, value = self.forward(obs)
        cache_action = cache_probs.squeeze().numpy()
        offload_action = int(offload_prob.item() > 0.5)
        return cache_action, offload_action, value.item()

    def remember(self, state, action, logprob, reward, value, done):
        self.buffer.append((state, action, logprob, reward, value, done))

    def compute_returns(self, rewards, values, dones):
        returns = []
        G = 0
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        return returns

    def train_step(self):
        if not self.buffer:
            return
        states, actions, logprobs_old, rewards, values, dones = zip(*self.buffer)

        returns = self.compute_returns(rewards, values, dones)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        logprobs_old = torch.tensor(np.array(logprobs_old), dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        # PPO training step
        self.hidden_state = None  # Reset hidden state for batch processing
        cache_probs, _, new_values = self.forward(states.unsqueeze(1))  # [B, 1, obs_dim]
        dist = torch.distributions.Categorical(cache_probs)
        logprobs_new = dist.log_prob(actions.argmax(dim=-1))

        ratios = torch.exp(logprobs_new - logprobs_old)
        advantages = returns - values
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.buffer.clear()

    def reset_hidden(self):
        self.hidden_state = None


if __name__ == "__main__":
    agent = GRUPPOAgent(obs_dim=28, num_contents=10)
    dummy_obs = np.random.rand(28).astype(np.float32)
    cache_action, offload_action, value = agent.act(dummy_obs)
    print("Cache probs:", cache_action)
    print("Offload decision:", "offload" if offload_action else "local")
    print("Estimated value:", value)
