import os
import shutil
import argparse
import numpy as np
import mujoco_py
import gym
from algorithm.ppo import PPO
from algorithm.policy import MLPGaussianPolicy
from algorithm.value_fn import MLPValueFn
from data_csv_saver import DataCSVSaver


def reset_dir(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


### Returns policy instance
def train(env, save_dir: str, epochs: int, update_epochs: int, agents: int, trajectory_steps: int) -> object:
    reset_dir(save_dir)

    policy = MLPGaussianPolicy(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        hidden_layers=[3, 2],
        action_high=env.action_space.high,
        action_low=env.action_space.low,
    )
    value_fn = MLPValueFn(env.observation_space.shape[0], hidden_layers=[3, 2])
    rl = PPO(policy, value_fn, update_epochs=update_epochs)

    reward_log = DataCSVSaver(os.path.join(save_dir, "distance.txt"), ("epoch", "averaged reward")) # log epoch development of reward
    loss_log = DataCSVSaver(os.path.join(save_dir, "loss.txt"), ("epoch", "iter", "loss")) # log loss transition to check whether network update is carried out properly

    e = 0
    # training
    while e < epochs:
        try:
            print("Epoch {} ......".format(e))
            rl.reset_sample_buffers()

            average_rewards = []

            # sampling
            print("  sampling...")
            for n in range(agents):
                # buffer to save samples from trajectory
                observations_list = []
                actions_list = []
                rewards_list = []

                # init
                obs = env.reset()
                observations_list.append(obs)

                # run a trajectory
                for t in range(trajectory_steps):
                    action = rl.act(obs)
                    obs, r, done, _ = env.step(action)
                    # save a sample
                    actions_list.append(action)
                    observations_list.append(obs)
                    rewards_list.append(r)
                    if done:
                        break
                
                rl.feed_trajectory_samples(observations_list, actions_list, rewards_list, done)

                print("    agent {}: run for {} steps, average reward {}".format(n, t, np.mean(rewards_list)))
                average_rewards.append(np.mean(rewards_list))

            # update parameters of policy and state value function
            print("  updating...")
            update_epoch_losses = rl.update()
            
            # logging
            reward_log.append_data(e, np.mean(average_rewards)) # save averaged reward
            for i, loss in enumerate(update_epoch_losses):
                loss_log.append_data(e, i, loss) # save loss of each update epoch

            print("  average reward {}".format(np.mean(average_rewards)))
            e += 1

        except KeyboardInterrupt:
            command = input("\nSample? Finish? : ")
            if command in ["sample", "Sample"]:
                # run for X steps
                sample_steps = input("How many steps for this sample?: ")
                if sample_steps == "":
                    sample_steps = steps
                    print("default steps {}".format(sample_steps))
                else:
                    sample_steps = int(sample_steps)

                obs = env.reset()
                acc_r = 0
                for t in range(sample_steps):
                    env.render()
                    action = rl.act(obs)
                    obs, r, done, _ = env.step(action)
                    acc_r += r

                continue
            if command in ["finish", "Finish"]:
                print("Ending training ...")
                break

    print("Finish training. Saving the policy and value_fn in {}".format(save_dir))
    rl.save(save_dir)
    return rl.policy
                

### Run and render
### Returns accumulated reward
def test(env, policy, steps: int, render=True) -> float:
    obs = env.reset()
    accumulated_reward = 0
    for i in range(steps):
        env.render()
        action = policy.act(obs)
        obs, reward, done, _ = env.step(action)
        accumulated_reward += reward

    return accumulated_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train pendulum')
    parser.add_argument('--epochs', type=int, default=100, help="epochs of training")
    parser.add_argument('--update_epochs', type=int, default=10, help="update iteration times during an epoch")
    parser.add_argument('--agents', type=int, default=10, help="number of independent agents during sampling")
    parser.add_argument('--steps', type=int, default=1000, help="steps of sampling")
    parser.add_argument('save_dir', help="path to a directory where the trained model is saved")
    args = parser.parse_args()

    print("Train pendulum")
    env = gym.make('Pendulum-v0')
    policy = train(env, args.save_dir, args.epochs, args.update_epochs, args.agents, args.steps)
    reward = test(env, policy, 200)
    print("Get accumulated reward {} after training {} epochs".format(reward, args.epochs))
    env.close()