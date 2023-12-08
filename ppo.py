import gymnasium as gym
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from collections import deque
import numpy as np
import argparse
import os

HYPERPARAMS = {
    'LunarLander-v2' : {
        'horizon': 4096,
        'total_timesteps': 4096 * 500,
        'no_of_batch_timesteps': 4096 * 4,
        'no_of_batch_episodes': 10,
        'no_of_episodes': 1000,
        'gamma' : 0.99,
        'solved_score' : 200,
        'policy_learning_rate' : 1e-3,
        'value_learning_rate' : 5e-3,
        'no_of_epochs': 10,
        'epsilon' : 0.2,
        'batch_size': 256,
    },

     'CartPole-v1' : {
        'horizon': 501,
        'total_timesteps': 100000,
        'no_of_batch_timesteps': 2000,
        'no_of_batch_episodes': 10,
        'no_of_episodes': 300,
        'gamma' : 0.99,
        'solved_score' : 200,
        'policy_learning_rate' : 1e-3,
        'value_learning_rate' : 5e-3,
        'no_of_epochs': 10,
        'epsilon' : 0.2,
        'batch_size': 32,
    },

     'Acrobot-v1' : {
        'horizon': 501,
        'total_timesteps': 100000,
        'no_of_batch_timesteps': 2000,
        'no_of_batch_episodes': 10,
        'no_of_episodes': 500,
        'gamma' : 0.99,
        'solved_score' : -100,
        'policy_learning_rate' : 1e-3,
        'value_learning_rate' : 5e-3,
        'no_of_epochs': 10,
        'epsilon' : 0.2,
        'batch_size': 256,
    },
}

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class RolloutDataset(Dataset):
    def __init__(self, states, actions, log_action_probs, rewards):
        self.states = states
        self.actions = actions
        self.log_action_probs = log_action_probs
        self.rewards = rewards

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.log_action_probs[index], self.rewards[index]
        
# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size1=128, hidden_size2=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = F.softmax(out, dim = 1)
        return out

# Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size1=128, hidden_size2=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
class PPO:
    def __init__(self, env, rand_seed=42, root_dir='ppo', eval_mode=False):
        self.env = env
        self.env_name = env.spec.id
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.hyper_params = HYPERPARAMS[self.env_name]
        self.root_dir = root_dir
        self.eval_mode = eval_mode
        
        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim)
        self.value_network = ValueNetwork(self.state_dim)
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=self.hyper_params['policy_learning_rate'])
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=self.hyper_params['value_learning_rate'])
        
        self.recent_episode_rewards = deque(maxlen=10)
        self.no_episodes = 0
        self.ep_rewards = []
        self.rand_seed = rand_seed
        self.time_steps = 0
        self.episode_lens = []
        self.model_dir = os.path.join(self.root_dir, self.env.spec.id, self.get_hyper_param_str())
        self.max_mean_reward_so_far = float('-inf')
        os.makedirs(self.model_dir, exist_ok=True)

    def get_hyper_param_str(self):
        hyper_param_str = ''
        for key, value in self.hyper_params.items():
            hyper_param_str += f'{key}={value}_'
        if hyper_param_str:
            hyper_param_str = hyper_param_str[:-1]
        return hyper_param_str

    def save_models(self):
        torch.save(self.value_network.state_dict(), os.path.join(self.model_dir, f'value_network_rand_seed={self.rand_seed}.pth'))
        torch.save(self.policy_network.state_dict(), os.path.join(self.model_dir, f'policy_network_rand_seed={self.rand_seed}.pth'))

    def save_arrays(self):
        np.save(os.path.join(self.model_dir, f'ep_rewards_rand_seed={self.rand_seed}.npy'), np.array(self.ep_rewards))
        np.save(os.path.join(self.model_dir, f'episode_lens_rand_seed={self.rand_seed}.npy'), np.array(self.episode_lens))

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_network(state)
        if self.eval_mode:
            action = probs.argmax()
            return action.item(), None
        dist = Categorical(probs)
        action = dist.sample()

        return action.detach().item(), dist.log_prob(action).detach().item()
    
    def get_discounted_rewards(self, batch_rewards):
        discounted_rewards = deque()
        for episode_rewards in batch_rewards[::-1]:
            cumulative_reward = 0
            for reward in episode_rewards[::-1]:
                cumulative_reward = reward + self.hyper_params['gamma'] * cumulative_reward
                discounted_rewards.appendleft(cumulative_reward)
        return discounted_rewards
    
    
    def rollout(self):
        no_of_batch_timesteps = self.hyper_params['no_of_batch_timesteps']
        horizon = self.hyper_params['horizon']
        num_batch_episodes = 0
        timesteps = 0
        batch_rewards = []
        batch_len = 0
        states = []
        actions = []
        log_action_probs = []
        while timesteps < no_of_batch_timesteps and num_batch_episodes < self.hyper_params['no_of_batch_episodes']:
            ep_reward = 0
            ep_rewards_arr = []
            state, _ = self.env.reset()
            ep_len = 0

            for _ in range(horizon):

                action, log_prob = self.get_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                ep_reward += reward
                states.append(state)
                actions.append(action)
                log_action_probs.append(log_prob)
                ep_rewards_arr.append(reward)
                state = next_state
                timesteps += 1
                self.time_steps += 1
                batch_len += 1
                ep_len += 1
                if done or truncated:
                    break
            num_batch_episodes += 1
            self.no_episodes += 1
            self.ep_rewards.append(ep_reward)
            batch_rewards.append(ep_rewards_arr)
            self.recent_episode_rewards.append(ep_reward)
            self.episode_lens.append(ep_len)
            if self.no_episodes % 10 == 0:
                print("Episode: {}, Recent Averaged Reward: {}, Episode reward: {}".format(self.no_episodes, np.mean(self.recent_episode_rewards), ep_reward))

            if np.mean(self.recent_episode_rewards) > self.max_mean_reward_so_far:
                self.max_mean_reward_so_far = np.mean(self.recent_episode_rewards)
                self.save_models()
        return states, actions, log_action_probs, batch_rewards, batch_len
    
    def get_action_probs_and_state_values(self, states_tensor, actions_tensor):
    
        action_probs = self.policy_network(states_tensor)
        dist = Categorical(action_probs)
        actions_tensor = actions_tensor.unsqueeze(1)
        action_probs = action_probs.gather(1, actions_tensor).squeeze()


        state_values = self.value_network(states_tensor).squeeze()
        return torch.log(action_probs), state_values, dist.entropy()


    def train(self):
        self.policy_network.train()
        self.value_network.train()
        torch.autograd.set_detect_anomaly(True)
        while self.time_steps < self.hyper_params['total_timesteps'] and self.no_episodes < self.hyper_params['no_of_episodes']:
            states, actions, log_action_probs, batch_rewards, batch_len = self.rollout()
            discounted_rewards = self.get_discounted_rewards(batch_rewards)

        
            old_action_log_probs = torch.tensor(log_action_probs)
            rollout_dataset = RolloutDataset(states, actions, old_action_log_probs, discounted_rewards)
            rollout_dataloader = DataLoader(rollout_dataset, batch_size=self.hyper_params['batch_size'], shuffle=True) 
            for epoch in range(self.hyper_params['no_of_epochs']):
                for batch_states, batch_actions, batch_old_action_log_probs, batch_target_rewards in rollout_dataloader:

                    batch_new_action_probs, batch_state_values, entropy = self.get_action_probs_and_state_values(batch_states, batch_actions)


                    ratios = torch.exp(batch_new_action_probs - batch_old_action_log_probs)
                    advantages = batch_target_rewards - batch_state_values.detach()
                    

                    # normalizing advantages
                    if advantages.numel() <= 1:
                        continue
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
                    surrogate_loss_1 = ratios * advantages
                    
                    surrogate_loss_2 = torch.clamp(ratios, 1 - self.hyper_params['epsilon'], 1 + self.hyper_params['epsilon']) * advantages

                    policy_loss = -torch.min(surrogate_loss_1, surrogate_loss_2).mean() - 0.01 * entropy.mean()

                   
                    value_loss = F.mse_loss(batch_target_rewards.float(), batch_state_values.float())

                    self.optimizer_policy.zero_grad()
                    policy_loss.backward()
                    self.optimizer_policy.step()

                    self.optimizer_value.zero_grad()
                    value_loss.backward()
                    self.optimizer_value.step()
        self.save_arrays()


    def eval(self, no_episodes=10):
        self.policy_network.load_state_dict(torch.load(os.path.join(self.model_dir, f'policy_network_rand_seed={self.rand_seed}.pth')))
        self.value_network.load_state_dict(torch.load(os.path.join(self.model_dir, f'value_network_rand_seed={self.rand_seed}.pth')))
        self.policy_network.eval()
        self.value_network.eval()
        with torch.no_grad():
            for ep in range(no_episodes):
                episode_reward = 0
                state, _ = self.env.reset()
                for _ in range(self.hyper_params['horizon']):
                    action, _ = self.get_action(state)
                    state, reward, done, truncated, _  = self.env.step(action)
                    episode_reward += reward
                    if done or truncated:
                        break
                print(f'Episode {ep} reward: {episode_reward}')


if __name__ == '__main__':
    eval_only=False
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--env_name', type=str, default='LunarLander-v2')
    parser.add_argument('--rand_seed', type=int, default=0)

    args = parser.parse_args()
    rand_seed = args.rand_seed
    set_seeds(rand_seed)
    eval_only = args.eval_only
    env_name = args.env_name

    print(f'Training environment {env_name} with random seed {rand_seed}.... \n\n')
    if not eval_only:
        env = gym.make(env_name)
        agent = PPO(env=env, rand_seed=rand_seed)
        agent.train()
        agent.env.close()

    print(f'Evaluating environment {env_name} with random seed {rand_seed}.... \n\n')
    env = gym.make(env_name, render_mode='human')
    agent = PPO(env=env, rand_seed=rand_seed, eval_mode=True)
    agent.eval()