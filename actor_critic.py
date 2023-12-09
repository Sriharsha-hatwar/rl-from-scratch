import gymnasium as gym
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from collections import deque
import numpy as np
import argparse
import os

# create environment
HYPERPARAMS = {
    'LunarLander-v2' : {
        'STATE_DIM': 8,
        'NO_ACTIONS' : 4,
        'HORIZON': 4096,
        'NO_EPISODES' : 5000,
        'GAMMA' : 0.99,
        'SOLVED_SCORE' : 200,
        'policy_learning_rate' : 1e-4,
        'value_learning_rate' : 1e-3,
        'decay_lr_with_timestep' : False
    },
    'Acrobot-v1' : {
        'STATE_DIM': 6,
        'NO_ACTIONS' : 3,
        'HORIZON': 501,
        'NO_EPISODES' : 250,
        'GAMMA' : 0.99,
        'SOLVED_SCORE' : -100,
        'policy_learning_rate' : 1e-4,
        'value_learning_rate' : 1e-3,
        'decay_lr_with_timestep' : True
    },
    'CartPole-v1' : {
        'STATE_DIM': 4,
        'NO_ACTIONS' : 2,
        'HORIZON': 501,
        'NO_EPISODES' : 700,
        'GAMMA' : 0.99,
        'SOLVED_SCORE' : 500,
        'policy_learning_rate' : 1e-4,
        'value_learning_rate' : 1e-3,
        'decay_lr_with_timestep' : True
    },
    
}

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_environment(env_name='LunarLander-v2', eval_only=False):
    env = gym.make(env_name,
        render_mode='human' if eval_only else None,
        # continuous=False,
        # gravity=-10.0,
        # enable_wind=False,
        # wind_power=0.0,
        # turbulence_power=0.0,
    )
    hyper_params = HYPERPARAMS[env_name]
    return env, hyper_params


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
    
def get_hyper_param_str(hyper_params):
    hyper_param_str = ''
    for key, value in hyper_params.items():
        hyper_param_str += f'{key}={value}_'
    if hyper_param_str:
        hyper_param_str = hyper_param_str[:-1]
    return hyper_param_str

# Initialize the models
def save_models(policy_network, value_network, model_dir='actor_critic/', rand_seed=0):
    torch.save(value_network.state_dict(), os.path.join(model_dir, f'value_network_rand_seed={rand_seed}.pth'))
    torch.save(policy_network.state_dict(), os.path.join(model_dir, f'policy_network_rand_seed={rand_seed}.pth'))

class Agent:
    def __init__(self, env_name='LunarLander-v2', eval_only=False, root_dir='actor_critic/', rand_seed=0):
        self.env_name = env_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rand_seed = rand_seed
        self.env, self.hyper_params = create_environment(env_name, eval_only)
        self.state_dim = self.hyper_params['STATE_DIM']
        self.action_dim = self.hyper_params['NO_ACTIONS']
        self.no_episodes = self.hyper_params['NO_EPISODES']
        self.gamma = self.hyper_params['GAMMA']
        self.solved_score = self.hyper_params['SOLVED_SCORE']
        self.horizon = self.hyper_params['HORIZON']
        self.decay_lr_with_timestep = self.hyper_params.get('decay_lr_with_timestep', False)
        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim, hidden_size1=128, hidden_size2=64).to(self.device)
        self.value_network = ValueNetwork(self.state_dim, hidden_size1=128, hidden_size2=64).to(self.device)
        if not eval_only:
            self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.hyper_params['policy_learning_rate'])
            self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.hyper_params['value_learning_rate'])
        self.eval_only = eval_only
        self.root_dir = root_dir
        self.max_so_far = -np.inf
        self.model_dir = os.path.join(self.root_dir, self.env_name, get_hyper_param_str(self.hyper_params))
        os.makedirs(self.model_dir, exist_ok=True)


    def select_action(self, state, eval_mode=False):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state_tensor)
        if not eval_mode:
            m = Categorical(action_probs)
            action = m.sample()
            return action.item(), m.log_prob(action)
        return torch.argmax(action_probs).item(), None
    
    def train(self):
        all_scores = []
        all_steps = []
        recent_scores = deque(maxlen=50)
        self.policy_network.train()
        self.value_network.train()
        for ep in range(self.no_episodes):
            steps = 0
            episode_reward = 0
            state, _ = self.env.reset()
            I = 1
            for _ in range(self.horizon):
                steps += 1
                # select action
                action, log_prob = self.select_action(state)
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                state_value = self.value_network(state_tensor)

                next_state, reward, done, truncated, _  = self.env.step(action)
                episode_reward += reward

                if done or truncated:
                    next_state_value = torch.tensor([0.0]).float().unsqueeze(0).to(self.device)
                else:
                    next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
                    next_state_value = self.value_network(next_state_tensor)


                one_step_return = reward + self.gamma * next_state_value

                value_loss = I * F.mse_loss(one_step_return, state_value)

                advantage = one_step_return.detach().item() - state_value.detach().item()
                policy_loss = (I * (-log_prob) * advantage)


                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                state = next_state
                if self.decay_lr_with_timestep:
                    I *= self.gamma
                if done or truncated:
                    break
            all_scores.append(episode_reward)
            recent_scores.append(episode_reward)
            all_steps.append(steps)
            last_ten_mean_score = np.array(recent_scores).mean()
            if ep % 10 == 0:
                print(f'Episode {ep} recent reward: {last_ten_mean_score}, episode reward: {episode_reward}')
    
                # np.save(os.path.join(self.model_dir, f'recent_scores_rand_seed={self.rand_seed}.npy'), np.array(recent_scores))
            if last_ten_mean_score > self.max_so_far:
                self.max_so_far = last_ten_mean_score
                save_models(self.policy_network, self.value_network, model_dir=self.model_dir, rand_seed=self.rand_seed)
            # if last_ten_mean_score >= self.solved_score:
            #     break
        np.save(os.path.join(self.model_dir, f'all_scores_rand_seed={self.rand_seed}.npy'), np.array(all_scores))
        np.save(os.path.join(self.model_dir, f'all_steps_rand_seed={self.rand_seed}.npy'), np.array(all_steps))
    def eval(self, no_episodes=10):
        self.policy_network.load_state_dict(torch.load(os.path.join(self.model_dir, f'policy_network_rand_seed={self.rand_seed}.pth')))
        self.value_network.load_state_dict(torch.load(os.path.join(self.model_dir, f'value_network_rand_seed={self.rand_seed}.pth')))
        self.policy_network.eval()
        self.value_network.eval()
        with torch.no_grad():
            for ep in range(no_episodes):
                episode_reward = 0
                state, _ = self.env.reset()
                for _ in range(self.horizon):
                    self.env.render()
                    action, _ = self.select_action(state, eval_mode=True)
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
    if not eval_only:
        agent = Agent(env_name=env_name, eval_only=False, root_dir='actor_critic/', rand_seed=rand_seed)
        agent.train()
        agent.env.close()
    agent = Agent(env_name=env_name, eval_only=True, root_dir='actor_critic/', rand_seed=rand_seed)
    agent.eval()