import torch
import torch.nn.functional as F
import numpy as np

from roboro.networks import CNN, MLP

class RepresentationNetwork(torch.nn.Module):
    def __init__(self, obs_shape, latent_size):
        super().__init__()
        self.feat_enc = CNN(obs_shape, latent_size) if len(obs_shape) == 3\
                else MLP(np.prod(obs_shape), latent_size)

    def forward(self, x):
        x = self.feat_enc(x)
        return x

class DynamicsNetwork(torch.nn.Module):
    def __init__(self, encoded_obs_size, action_shape, hidden_size=128):
        super().__init__()
        self.action_shape = action_shape
        self.mlp = MLP(encoded_obs_size + action_shape, hidden_size)
        self.r_head = torch.nn.Linear(hidden_size, 1)
        self.h_obs_head = torch.nn.Linear(hidden_size, encoded_obs_size)

    def forward(self, obs_h, action):
        onehot = torch.zeros(size=(len(action),self.action_shape)).type_as(action)
        onehot.scatter_(1, action.long(), 1.0)
        x = torch.cat([obs_h, onehot], dim=-1)
        x = self.mlp(x)
        reward = self.r_head(x)
        next_obs_h = self.h_obs_head(x)
        return next_obs_h, reward

class PredictionNetwork(torch.nn.Module):
    def __init__(self, encoded_state_dim, action_shape, hidden_size=128):
        super().__init__()
        self.mlp = MLP(encoded_state_dim, hidden_size)
        self.policy_head = MLP(hidden_size, action_shape)
        self.value_head = MLP(hidden_size, 1)

    def forward(self, x):
        h =  F.relu(self.mlp(x))
        policy_q_val = self.policy_head(h)
        value = self.value_head(h)
        return policy_q_val, value

class MuZero(torch.nn.Module):
    def __init__(self, obs_shape, action_shape, normalizer, obs_h_dim=10):
        super().__init__()
        self.normalizer = normalizer
        self.criterion = torch.nn.MSELoss()
        self.action_shape = action_shape
        self.obs_shape = obs_shape

        self.representation_model = RepresentationNetwork(obs_shape, obs_h_dim).float()
        self.dynamics_model = DynamicsNetwork(obs_h_dim, action_shape).float()
        self.prediction_model = PredictionNetwork(obs_h_dim, action_shape).float()

        self.optim = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-4)

    def forward(self, obs, action, next_obs):
        obs_h = self.representation_model(obs)
        next_obs_h, pred_reward =  self.dynamics_model(torch.cat([obs_h, action], dim=-1))
        policy_a, value = self.prediction_model(obs_h)

    @torch.no_grad()
    def predict_steps(self, obs, n=50):
        self.eval()
        obs = self.normalizer.norm(obs)
        obs_h = self.representation_model(obs)
        action_dict = {}
        for init_a in range(self.action_shape):
            obs_h_next_pred, pred_reward =  self.dynamics_model(obs_h, torch.tensor(init_a).view(-1, 1).cuda())
            for _ in range(n):
                policy_q_val, value = self.prediction_model(obs_h_next_pred)
                policy_a = torch.argmax(policy_q_val, dim=1).unsqueeze(1)
                obs_h_next_pred, pred_reward =  self.dynamics_model(obs_h_next_pred, policy_a)
            action_dict[init_a] = value.flatten().item()

        #  print(action_dict)
        best_action = torch.tensor(max(action_dict, key=action_dict.get))

        return best_action.cpu().flatten(), value

    def update(self, buffer):
        self.train()
        losses = {}
        def get_batch():
            buffer.wm = True
            s_batch, a_batch, r_batch, s_next_batch, done_batch, values_batch = [], [], [], [], [], []
            for _ in range(128):
                bunch = next(iter(buffer))
                s_batch.append(bunch[0])
                a_batch.append(bunch[1])
                r_batch.append(bunch[2])
                done_batch.append(bunch[3])
                s_next_batch.append(bunch[4])
                values_batch.append(bunch[6])
            buffer.wm = False
            return torch.stack(s_batch).cuda(), torch.tensor(a_batch).unsqueeze(1).cuda(), torch.tensor(r_batch).unsqueeze(1).cuda(), torch.stack(s_next_batch).cuda(), torch.tensor(done_batch).unsqueeze(1).cuda(), torch.stack(values_batch).unsqueeze(1).cuda().detach()

        for _ in range(1000):
            self.optim.zero_grad()
            obs, action, reward, obs_next, done, v_batch, = get_batch()
            value_batch = v_batch.squeeze().mean(1, keepdim=True)
            q_val_batch = v_batch.squeeze()

            obs = self.normalizer.norm(obs)
            obs_h = self.representation_model(obs)
            obs_next = self.normalizer.norm(obs_next)
            obs_h_next = self.representation_model(obs_next)

            obs_h_next_pred, pred_reward =  self.dynamics_model(obs_h, action)

            obs_h_next_loss = self.criterion(obs_h_next_pred, obs_h_next)
            losses.setdefault("obs_h_next_loss", []).append(obs_h_next_loss.item())
            reward_loss = self.criterion(pred_reward, reward)
            losses.setdefault("reward_loss", []).append(reward_loss.item())

            policy_q_val, value = self.prediction_model(obs_h)
            value_loss = self.criterion(value, value_batch)
            losses.setdefault("value_loss", []).append(value_loss.item())
            policy_loss = self.criterion(policy_q_val, q_val_batch)
            losses.setdefault("policy_loss", []).append(policy_loss.item())

            loss = obs_h_next_loss + reward_loss + value_loss * 0.25 + policy_loss
            loss.backward()
            self.optim.step()

        for l_key in losses:
            losses[l_key] = np.mean(losses[l_key])

        return losses

