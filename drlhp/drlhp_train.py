from envs.gym_wrapper import *
from stable_baselines3.common.vec_env import SubprocVecEnv
from moral.ppo import *
from tqdm import tqdm
import torch
from preference_model import *
from moral.preference_giver import *
import wandb
import pickle
import argparse


if __name__ == '__main__':

    # Fetch ratio args
    parser = argparse.ArgumentParser(description='Preference Ratio.')
    parser.add_argument('--ratio', nargs='+', type=int)
    args = parser.parse_args()

    # Config
    wandb.init(project='PbRL', config={
        'env_id': 'randomized_v3',
        'ratio': args.ratio,
        'env_steps': 12e6,
        'batchsize_ppo': 12,
        'batchsize_preference': 12,
        'n_queries': 5000,
        'update_reward_freq': 50,
        'preference_warmup': 1,
        'pretrain': 1000,
        'n_workers': 12,
        'lr_ppo': 3e-4,
        'lr_reward': 3e-5,
        'entropy_reg': 1,
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5
    })
    config = wandb.config
    env_steps = int(config.env_steps / config.n_workers)
    query_freq = int(env_steps / (config.n_queries + 2))

    # Create Environment
    vec_env = SubprocVecEnv([make_env(config.env_id, i) for i in range(config.n_workers)])
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)
    preference_model = PreferenceModelConv(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(
        device)
    # preference_model.load_state_dict(torch.load('../saved_models/preference_model_v2.pt'))
    # preference_buffer = pickle.load(open('../saved_models/preference_buffer_v2.pk', 'rb'))
    preference_buffer = PreferenceBuffer()
    preference_optimizer = torch.optim.Adam(preference_model.parameters(), lr=config.lr_reward)
    preference_giver = TargetGiverv3(target=config.ratio)

    for t in tqdm(range(int(config.env_steps / config.n_workers))):
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = vec_env.step(actions)
        preference_state = torch.tensor(states).to(device).float()
        preference_rewards = preference_model.forward(preference_state, actions).squeeze(1)
        preference_rewards = list(preference_rewards.detach().cpu().numpy())

        train_ready = dataset.write_tuple(states, actions, preference_rewards, done, log_probs, rewards)

        if train_ready:
            objective_logs = dataset.log_objectives()
            for i in range(objective_logs.shape[1]):
                wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})
            for ret in dataset.log_returns():
                wandb.log({'Returns': ret})

            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)

            # Sample random pair of trajectories
            rand_idx = np.random.choice(range(len(dataset.trajectories)), 2, replace=False)
            tau_1 = dataset.trajectories[rand_idx[0]]
            tau_2 = dataset.trajectories[rand_idx[1]]

            dataset.reset_trajectories()

        if t % query_freq == 0 and t > config.pretrain:
            # Query user for latest pair
            logs_tau_1 = np.array(tau_1['logs']).sum(axis=0)
            logs_tau_2 = np.array(tau_2['logs']).sum(axis=0)
            print(f'Found trajectory pair: {logs_tau_1, logs_tau_2}')
            auto_preference = preference_giver.query_pair(logs_tau_1, logs_tau_2)
            # auto_preference = v2_soft_preference(logs_tau_1, logs_tau_2, threshold=5)
            print(auto_preference)

            preference_buffer.add_preference(tau_1, tau_2, auto_preference)

        if t % config.update_reward_freq == 0 and t > config.pretrain and \
                len(preference_buffer.storage) > config.preference_warmup:
            # Update preference model
            preference_loss = update_preference_model(preference_model, preference_buffer, preference_optimizer,
                                                      config.batchsize_preference)
            wandb.log({'Preference Loss': preference_loss})

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)
