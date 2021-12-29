from tqdm import tqdm
from ppo import PPO, TrajectoryDataset, update_policy
import torch
from airl import *
from active_learning import *
import numpy as np
import matplotlib.pyplot as plt
from envs.gym_wrapper import *
from preference_giver import *
from utils.evaluate_ppo import evaluate_ppo
import wandb
import argparse


# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    # Fetch ratio args for automatic preferences
    parser = argparse.ArgumentParser(description='Preference Ratio.')
    parser.add_argument('--ratio', nargs='+', type=int)
    args = parser.parse_args()

    # Config
    wandb.init(project='MORAL', config={
        'env_id': 'randomized_v3',
        'ratio': args.ratio,
        'env_steps': 8e6,
        'batchsize_ppo': 12,
        'n_queries': 50,
        'preference_noise': 0,
        'n_workers': 12,
        'lr_ppo': 3e-4,
        'entropy_reg': 0.25,
        'gamma': 0.999,
        'epsilon': 0.1,
        'ppo_epochs': 5
    })
    config = wandb.config
    env_steps = int(config.env_steps/config.n_workers)
    query_freq = int(env_steps/(config.n_queries+2))

    # Create Environment
    vec_env = VecEnv(config.env_id, config.n_workers)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    print('Initializing and Normalizing Rewards...')
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)

    # Expert 0
    discriminator_0 = Discriminator(state_shape=state_shape, in_channels=in_channels).to(device)
    discriminator_0.load_state_dict(torch.load('../saved_models/discriminator_v3_[0,1,0,1].pt'))
    ppo_0 = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    ppo_0.load_state_dict(torch.load('../saved_models/ppo_airl_v3_[0,1,0,1].pt'))
    utop_0 = discriminator_0.estimate_utopia(ppo_0, config)
    print(f'Reward Normalization 0: {utop_0}')
    discriminator_0.set_eval()

    # Expert 1
    discriminator_1 = Discriminator(state_shape=state_shape).to(device)
    discriminator_1.load_state_dict(torch.load('../saved_models/discriminator_v3_[0,0,1,1].pt'))
    ppo_1 = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    ppo_1.load_state_dict(torch.load('../saved_models/ppo_airl_v3_[0,0,1,1].pt'))
    utop_1 = discriminator_1.estimate_utopia(ppo_1, config)
    print(f'Reward Normalization 1: {utop_1}')
    discriminator_1.set_eval()


    dataset = TrajectoryDataset(batch_size=config.batchsize_ppo, n_workers=config.n_workers)

    # Logging
    objective_logs = []
    checkpoint_logs = []

    # Active Learning
    preference_learner = PreferenceLearner(d=len(config.ratio), n_iter=10000, warmup=1000)
    w_posterior = preference_learner.sample_w_prior(preference_learner.n_iter)
    w_posterior_mean = w_posterior.mean(axis=0)
    volume_buffer = VolumeBuffer()
    preference_giver = PreferenceGiverv3(ratio=config.ratio)

    for t in tqdm(range(env_steps)):

        # Query User
        if t % query_freq == 0 and t > 0:
            best_delta = volume_buffer.best_delta

            # Using ground truth returns for preference elicitation
            ret_a, ret_b = volume_buffer.best_returns
            print(f'Found trajectory pair: {(ret_a, ret_b)}')
            print(f'Corresponding best delta: {best_delta}')
            preference = preference_giver.query_pair(ret_a, ret_b)
            print(f'obtained preference: {preference}')

            # v2-Environment comparison
            #ppl_saved_a = ret_a[1]
            #goal_time_a = ret_a[0]
            #ppl_saved_b = ret_b[1]
            #goal_time_b = ret_b[0]
            #if np.random.rand() < config.preference_noise:
            #    rand_pref_param = np.random.rand()
            #    if rand_pref_param > 0.5:
            #        preference = 1
            #    else:
            #        preference = -1
            #else:
            #    if ppl_saved_a > ppl_saved_b:
            #        preference = 1
            #    elif ppl_saved_b > ppl_saved_a:
            #        preference = -1
            #    elif goal_time_a > goal_time_b:
            #        preference = 1
            #    elif goal_time_b > goal_time_a:
            #        preference = -1
            #    else:
            #        preference = 1 if np.random.rand() < 0.5 else -1

            # Run MCMC
            preference_learner.log_preference(best_delta, preference)
            w_posterior = preference_learner.mcmc_vanilla()
            w_posterior_mean = w_posterior.mean(axis=0)
            w_posterior_mean = w_posterior_mean/np.linalg.norm(w_posterior_mean)
            print(f'Posterior Mean {w_posterior_mean}')

            volume_buffer.reset()

        # Environment interaction
        actions, log_probs = ppo.act(states_tensor)
        next_states, rewards, done, info = vec_env.step(actions)
        objective_logs.append(rewards)

        # Fetch AIRL rewards
        airl_state = torch.tensor(states).to(device).float()
        airl_next_state = torch.tensor(next_states).to(device).float()
        airl_rewards_0 = discriminator_0.forward(airl_state, airl_next_state, config.gamma).squeeze(1)
        airl_rewards_1 = discriminator_1.forward(airl_state, airl_next_state, config.gamma).squeeze(1)
        airl_rewards_0 = airl_rewards_0.detach().cpu().numpy() * [0 if i else 1 for i in done]
        airl_rewards_1 = airl_rewards_1.detach().cpu().numpy() * [0 if i else 1 for i in done]
        vectorized_rewards = [[r[0], airl_rewards_0[i], airl_rewards_1[i]] for i, r in enumerate(rewards)]
        scalarized_rewards = [np.dot(w_posterior_mean, r[0:3]) for r in vectorized_rewards]

        # v2-Environment
        #vectorized_rewards = [[r[0], airl_rewards_0[i]] for i, r in enumerate(rewards)]
        #scalarized_rewards = [np.dot(w_posterior_mean, r) for r in vectorized_rewards]

        # Logging obtained rewards for active learning
        volume_buffer.log_rewards(vectorized_rewards)
        # Logging true objectives for automatic preferences
        volume_buffer.log_statistics(rewards)
        # Add experience to PPO dataset
        train_ready = dataset.write_tuple(states, actions, scalarized_rewards, done, log_probs)

        if train_ready:
            # Log Objectives
            objective_logs = np.array(objective_logs).sum(axis=0)
            for i in range(objective_logs.shape[1]):
                wandb.log({'Obj_' + str(i): objective_logs[:, i].mean()})
            objective_logs = []

            # Update Models
            update_policy(ppo, dataset, optimizer, config.gamma, config.epsilon, config.ppo_epochs,
                          entropy_reg=config.entropy_reg)
            for ret in dataset.log_returns():
                wandb.log({'Returns': ret})

            # Sample two random trajectories & compare expected volume removal with best pair
            if volume_buffer.auto_pref:
                new_returns_a, new_returns_b, logs_a, logs_b = volume_buffer.sample_return_pair()
                volume_buffer.compare_delta(w_posterior, new_returns_a, new_returns_b, logs_a, logs_b, random=False)
            else:
                new_returns_a, new_returns_b = volume_buffer.sample_return_pair()
                volume_buffer.compare_delta(w_posterior, new_returns_a, new_returns_b)

            # Reset PPO buffer
            dataset.reset_trajectories()

        # Prepare state input for next time step
        states = next_states.copy()
        states_tensor = torch.tensor(states).float().to(device)
