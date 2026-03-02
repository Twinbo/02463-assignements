#Denne kode er blevet inspireret af vores underviseres eksempel, som vi har videre bygget på og tilpasset til vores eget projekt.


# %% Load libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from Flappybird import FlappyBirdGame
import pygame
import os

import random
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
os.environ["PYTHONHASHSEED"] = str(SEED)

def seed_all(seed: int):
    import os, random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    
def greedy_eval_stats(q_net, n_eval_games: int, eval_seed: int):
    seed_all(eval_seed)

    q_net.eval()
    eval_env = FlappyBirdGame()

    eval_scores = []
    eval_pipe_scores = []
    eval_steps = []

    with torch.no_grad():
        for _ in range(n_eval_games):
            done = False
            obs = state_to_input(eval_env.reset_game())

            ep_score = 0.0
            ep_steps = 0
            ep_pipe_score = 0

            while (not done) and (ep_steps < max_episode_step):
                action = int(torch.argmax(q_net(obs)).item())  # greedy
                obs_next, reward, done, score_game_pipe = eval_env.step(action)
                obs = state_to_input(obs_next)

                ep_score += float(reward)
                ep_pipe_score = score_game_pipe
                ep_steps += 1

            eval_scores.append(ep_score)
            eval_pipe_scores.append(ep_pipe_score)
            eval_steps.append(ep_steps)

    eval_env.close()

    return {
        "seed": eval_seed,
        "scores": np.array(eval_scores, dtype=np.float32),
        "pipes": np.array(eval_pipe_scores, dtype=np.float32),
        "steps": np.array(eval_steps, dtype=np.int32),
        "pipes_mean": float(np.mean(eval_pipe_scores)),
        "pipes_std": float(np.std(eval_pipe_scores)),
        "scores_mean": float(np.mean(eval_scores)),
        "scores_std": float(np.std(eval_scores)),
        "steps_mean": float(np.mean(eval_steps)),
    }


def greedy_eval_two_seed_mean(q_net, n_eval_games: int, trial_seed: int,
                             offset1: int = 12345, offset2: int = 22345):
    s1 = trial_seed + offset1
    s2 = trial_seed + offset2

    r1 = greedy_eval_stats(q_net, n_eval_games=n_eval_games, eval_seed=s1)
    r2 = greedy_eval_stats(q_net, n_eval_games=n_eval_games, eval_seed=s2)

    pipes_all = np.concatenate([r1["pipes"], r2["pipes"]])
    scores_all = np.concatenate([r1["scores"], r2["scores"]])
    steps_all = np.concatenate([r1["steps"], r2["steps"]])

    return {
        "seed1": s1,
        "seed2": s2,
        "r1": r1,
        "r2": r2,
        "pipes_all": pipes_all,
        "scores_all": scores_all,
        "steps_all": steps_all,
        "pipes_mean": float(np.mean(pipes_all)),
        "pipes_std": float(np.std(pipes_all)),
        "scores_mean": float(np.mean(scores_all)),
        "scores_std": float(np.std(scores_all)),
        "steps_mean": float(np.mean(steps_all)),
    }
    
# %% Parameter
n_games = 3000
epsilon = 1
epsilon_min = 0.001
epsilon_reduction_factor = 0.9995
gamma = 0.99
batch_size = 512
buffer_size = 750000
learning_rate = 0.0001
steps_per_gradient_update = 10
max_episode_step = 8000
input_dimension = 5
hidden_dimension = 256
output_dimension = 2
max_score = -40

learning_rate=2.645510e-04
epsilon_reduction_factor=0.974099
gamma=0.9995
hidden_dimension=512

# %% Neural network, optimizer and loss
q_net = torch.nn.Sequential(
    torch.nn.Linear(input_dimension, hidden_dimension),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dimension, hidden_dimension),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dimension, output_dimension)
)
optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
loss_function = torch.nn.MSELoss()

# %% State to input transformation
# Convert environment state to neural network input 
def state_to_input(state):
    bird_y, bird_vel, bird_dist, pipe_top, pipe_btm = state

    max_bird_y = 768
    max_dist = 400
    max_gap_height = 768


    # Normalize each value to the range [-1, 1]
    bird_y_normalized = (2 * bird_y / max_bird_y) - 1  
    bird_vel_normalized = bird_vel / 10  
    bird_dist_normalized = (2 * bird_dist / max_dist) - 1
    pipe_top_normal = (2 * pipe_top / max_gap_height) - 1
    pipe_btm_normal = (2 * pipe_btm / max_gap_height) - 1

     # Create a tensor with normalized values
    return torch.hstack((
        torch.tensor([bird_y_normalized], dtype=torch.float32),
        torch.tensor([bird_vel_normalized], dtype=torch.float32),
        torch.tensor([bird_dist_normalized], dtype=torch.float32),
        torch.tensor([pipe_top_normal], dtype=torch.float32),
        torch.tensor([pipe_btm_normal], dtype=torch.float32)
    ))

# %% Load the trained model from the checkpoint
# checkpoint_path = '/Users/haseebshafi/Desktop/Flappy bird/Q_net_checkpoints/q_net_checkpoint_100k_træning copy.pt'
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path)
#     q_net.load_state_dict(checkpoint['model_state_dict'])
#     print("Trained model loaded for testing.")
# else:
#     raise FileNotFoundError("Checkpoint not found. Train the model first.")


# %% Environment
env = FlappyBirdGame()
action_names = env.actions        # Actions the environment expects
actions = np.arange(2)            # Action numbers


# %% Buffers
obs_buffer = torch.zeros((buffer_size, input_dimension))
obs_next_buffer = torch.zeros((buffer_size, input_dimension))
action_buffer = torch.zeros(buffer_size).long()
reward_buffer = torch.zeros(buffer_size)
done_buffer = torch.zeros(buffer_size)


# %% check if data is available
#Load checkpoint if it exists 

#checkpoint_path = '/Users/haseebshafi/Desktop/Flappy bird/Q_net_checkpoints/q_net_checkpoint_100k_træning copy.pt'
checkpoint_path = '/Users/haseebshafi/Desktop/Flappy bird/Final 2/q_net_checkpoint_BO_5000.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    q_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step_count = checkpoint['step_count']
    epsilon = checkpoint['epsilon']
    print(f"Checkpoint loaded. Resuming training from Step {step_count}, Epsilon={epsilon:.4f}.")
else:
    step_count = 0
    epsilon = 1
    print("No checkpoint found. Starting training from scratch.")

# %% Training loop
# Logging
scores = []
losses = []
episode_steps = []
step_count = 0
print_interval = 100
pipe_scores = []
max_pipes_so_far = 0


# Training loop
for i in range(n_games):

    # Reset game
    score = 0
    episode_pipe_score = 0
    episode_step = 0
    episode_loss = 0
    episode_gradient_step = 0
    done = False
    env_observation = env.reset_game()
    observation = state_to_input(env_observation)

    # Reduce exploration rate
    epsilon = (epsilon-epsilon_min)*epsilon_reduction_factor + epsilon_min
    
    # Episode loop
    while (not done) and (episode_step < max_episode_step): 
        # Choose action and step environment
        if np.random.rand() < epsilon:
            # Random action
            action = np.random.choice(actions)
        else:
            # Action according to policy
            action = np.argmax(q_net(observation).detach().numpy())
        env_observation_next, reward, done, score_game_pipe = env.step(action)
        observation_next = state_to_input(env_observation_next)

        score += reward
        
        episode_pipe_score = score_game_pipe

        # Store to buffers
        buffer_index = step_count % buffer_size
        obs_buffer[buffer_index] = observation
        obs_next_buffer[buffer_index] = observation_next
        action_buffer[buffer_index] = action
        reward_buffer[buffer_index] = reward
        done_buffer[buffer_index] = done

        # Update to next observation
        observation = observation_next

        # Learn using minibatch from buffer (every steps_per_gradient_update)
        if step_count > batch_size and step_count%steps_per_gradient_update==0:
            # Choose a minibatch            
            batch_idx = np.random.choice(np.minimum(
                buffer_size, step_count), size=batch_size, replace=False)

            # Compute loss function
            out = q_net(obs_buffer[batch_idx])
            val = out[np.arange(batch_size), action_buffer[batch_idx]]  
            with torch.no_grad():
                out_next = q_net(obs_next_buffer[batch_idx])
                target = reward_buffer[batch_idx] + \
                    gamma*torch.max(out_next, dim=1).values * \
                    (1-done_buffer[batch_idx])
            loss = loss_function(val, target)

            # Step the optimizer
            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=20)

            optimizer.step()

            episode_gradient_step += 1
            episode_loss += loss.item()

        # Update step counteres
        episode_step += 1
        step_count += 1

    scores.append(score)
    losses.append(episode_loss / (episode_gradient_step+1))
    pipe_scores.append(episode_pipe_score)
    max_pipes_so_far = max(max_pipes_so_far, episode_pipe_score)
    episode_steps.append(episode_step)
    if (i+1) % print_interval == 0:
        std_score = np.std(scores)
        mean_score = np.mean(scores)
        average_score = np.mean(scores[-print_interval:])
        average_episode_steps = np.mean(episode_steps[-print_interval:])
        max_pipe_score = max(pipe_scores[-print_interval:])
        avg_pipe_score = np.mean(pipe_scores[-print_interval:])
        if np.max(scores[-print_interval:-1]) > max_score:
            max_score = np.max(scores[-print_interval:-1])
        
        # # Plot number of steps
        # plt.figure('Steps per episode')
        # plt.clf()
        # plt.plot(episode_steps, '.')
        # plt.xlabel('Episode')
        # plt.ylabel('Steps')
        # plt.grid(True)

        # # Plot average scores
        # average_score_window = 100 
        # if len(scores) >= average_score_window:
        #     avg_scores = np.convolve(scores, np.ones(average_score_window) / average_score_window, mode='valid')
        #     plt.figure('Average Score')
        #     plt.clf()
        #     plt.plot(range(len(avg_scores)), avg_scores, label=f'Average Score)', color='orange')
        #     plt.title(f'Step {step_count}: eps={epsilon:.3}')
        #     plt.xlabel('Episode')
        #     plt.ylabel('Average Score')
        #     plt.grid(True)
        #     plt.legend()

        # # Plot scores        
        # plt.figure('Score')
        # plt.clf()
        # plt.plot(scores, '.')
        # plt.title(f'Step {step_count}: eps={epsilon:.3}')
        # plt.xlabel('Episode')
        # plt.ylabel('Score')
        # plt.grid(True)

        # # Plot last batch loss
        # plt.figure('Loss')
        # plt.clf()
        # plt.plot(losses, '.')
        # plt.xlabel('Episode')
        # plt.ylabel('Loss')
        # plt.grid(True)

        # plt.pause(0.01)
        # #print(f'Episode={i+1}, Score={average_score:.1f}, mean={mean_score:.3f}, Steps={average_episode_steps:.0f}, max_score={max_score:.1f}, std_score={std_score:.1f}, Pipe_score{last_score_game}')
        
                # --- One window with 4 subplots ---
        # plt.ion()  # (valgfrit) sørger for at figuren opdaterer løbende

        # fig, axs = plt.subplots(2, 2, num="Training metrics", figsize=(12, 8))
        # fig.clf()  # ryd hele figuren
        # axs = fig.subplots(2, 2)  # genskab aksler efter clf()

        # ax_steps = axs[0, 0]
        # ax_avg   = axs[0, 1]
        # ax_score = axs[1, 0]
        # ax_loss  = axs[1, 1]

        # fig.suptitle(f"Step {step_count}: eps={epsilon:.3f}")

        # # 1) Steps per episode
        # ax_steps.plot(episode_steps, '.', markersize=2)
        # ax_steps.set_title("Steps per episode")
        # ax_steps.set_xlabel("Episode")
        # ax_steps.set_ylabel("Steps")
        # ax_steps.grid(True)

        # # 2) Average score (moving average)
        # average_score_window = 100
        # if len(scores) >= average_score_window:
        #     avg_scores = np.convolve(
        #         scores,
        #         np.ones(average_score_window) / average_score_window,
        #         mode='valid'
        #     )
        #     ax_avg.plot(range(len(avg_scores)), avg_scores, label="Moving avg (100)")
        #     ax_avg.legend()
        # ax_avg.set_title("Average score")
        # ax_avg.set_xlabel("Episode")
        # ax_avg.set_ylabel("Avg score")
        # ax_avg.grid(True)

        # # 3) Score
        # ax_score.plot(scores, '.', markersize=2)
        # ax_score.set_title("Score")
        # ax_score.set_xlabel("Episode")
        # ax_score.set_ylabel("Score")
        # ax_score.grid(True)

        # # 4) Loss
        # ax_loss.plot(losses, '.', markersize=2)
        # ax_loss.set_title("Loss")
        # ax_loss.set_xlabel("Episode")
        # ax_loss.set_ylabel("Loss")
        # ax_loss.grid(True)

        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # plads til suptitle
        # fig.canvas.draw()
        # plt.pause(0.01)

        print(f'Episode={i+1}, '
            f'Score={average_score:.1f}, '
            f'mean={mean_score:.3f}, '
            f'Steps={average_episode_steps:.0f}, '
            f'max_score={max_score:.1f}, '
            f'std_score={std_score:.1f}, '
            f'Max_Pipes={max_pipe_score}, '
            f'Avg_Pipes={avg_pipe_score:.2f}')
        
        # Save model and optimizer state as checkpoint
        torch.save({
            'model_state_dict': q_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step_count': step_count,
            'epsilon': epsilon
        }, 'q_net_checkpoint.pt')
        
        
# n_eval_games = 100
# q_net.eval()

# eval_scores = []
# eval_pipe_scores = []
# eval_steps = []

# EVAL_SEED = SEED + 12345
# seed_all(EVAL_SEED)

# # Optional: use a fresh environment (recommended)
# eval_env = FlappyBirdGame()


# with torch.no_grad():
#     for ep in range(n_eval_games):
#         done = False
#         obs = state_to_input(eval_env.reset_game())

#         ep_score = 0.0
#         ep_steps = 0
#         ep_pipe_score = 0

#         while (not done) and (ep_steps < max_episode_step):
#             # Greedy action (policy only)
#             q_values = q_net(obs)
#             action = int(torch.argmax(q_values).item())

#             obs_next, reward, done, score_game_pipe = eval_env.step(action)
#             obs = state_to_input(obs_next)

#             ep_score += float(reward)
#             ep_pipe_score = score_game_pipe
#             ep_steps += 1

#         eval_scores.append(ep_score)
#         eval_pipe_scores.append(ep_pipe_score)
#         eval_steps.append(ep_steps)
        
# eval_env.close()
# #q_net.train()  #if train after

# print("\n--- Evaluation results (greedy policy, no exploration) ---")
# print(f"Eval games: {n_eval_games}")
# print(f"Score: mean={np.mean(eval_scores):.2f}, std={np.std(eval_scores):.2f}, "
#       f"min={np.min(eval_scores):.2f}, max={np.max(eval_scores):.2f}")
# print(f"Pipes: mean={np.mean(eval_pipe_scores):.2f}, std={np.std(eval_pipe_scores):.2f}, "
#       f"min={np.min(eval_pipe_scores):.2f}, max={np.max(eval_pipe_scores):.2f}")
# print(f"Steps: mean={np.mean(eval_steps):.1f}")        


n_eval_games = 100
trial_seed: int = 42

eval_out = greedy_eval_two_seed_mean(q_net, n_eval_games=n_eval_games, trial_seed=trial_seed)

print("\n--- Evaluation results (greedy policy, BO-style 2-seed mean) ---")
print(f"Eval games per seed: {n_eval_games} (total {2*n_eval_games})")
print(f"Seeds: {eval_out['seed1']} and {eval_out['seed2']}")
print(f"Pipes: mean={eval_out['pipes_mean']:.2f}, std={eval_out['pipes_std']:.2f}, "
      f"min={np.min(eval_out['pipes_all']):.2f}, max={np.max(eval_out['pipes_all']):.2f}")
print(f"Score: mean={eval_out['scores_mean']:.2f}, std={eval_out['scores_std']:.2f}, "
      f"min={np.min(eval_out['scores_all']):.2f}, max={np.max(eval_out['scores_all']):.2f}")
print(f"Steps: mean={eval_out['steps_mean']:.1f}")

final_fig, axs = plt.subplots(3, 2, figsize=(20, 10), num="Final metrics (6 plots)")

ax_steps  = axs[0, 0]
ax_avg    = axs[0, 1]
ax_score  = axs[1, 0]
ax_loss   = axs[1, 1]
ax_pipes  = axs[2, 0]
ax_reward = axs[2, 1]

final_fig.suptitle(
    f"Final metrics | train={n_games} | eval={n_eval_games} | step={step_count} | eps={epsilon:.4f}"
)

# 1) Reward (train)
ax_score.plot(scores, ".", markersize=4, color="red")
ax_score.set_title("Reward (train)")
ax_score.set_xlabel("Episode")
ax_score.set_ylabel("Reward")
ax_score.grid(True)

# 2) Average score (moving average of train scores)
average_score_window = 100
if len(scores) >= average_score_window:
    avg_scores = np.convolve(scores, np.ones(average_score_window)/average_score_window, mode="valid")
    ax_avg.plot(avg_scores, color="red")
ax_avg.set_title(f"Average score (window={average_score_window})")
ax_avg.set_xlabel("Episode")
ax_avg.set_ylabel("Avg score")
ax_avg.grid(True)

# 3) Steps per episode
ax_steps.plot(episode_steps, ".", markersize=4, color="red")
ax_steps.set_title("Steps per episode")
ax_steps.set_xlabel("Episode")
ax_steps.set_ylabel("Steps")
ax_steps.grid(True)

# 4) Loss (train)
ax_loss.plot(losses, color="red")
ax_loss.set_title("Loss (train)")
ax_loss.set_xlabel("Episode")
ax_loss.set_ylabel("Loss")
ax_loss.grid(True)

# 5) Pipe score (evla)
ax_pipes.plot(eval_out['pipes_all'],".", markersize=4, color="royalblue")
ax_pipes.set_title(f"Pipes (eval, greedy policy), mean={np.mean(eval_out['pipes_all']):.2f}")
ax_pipes.set_xlabel("Eval episode")
ax_pipes.set_ylabel("Pipes")
ax_pipes.grid(True)

# 6) Reward score (eval)  (uses greedy policy results)
ax_reward.plot(eval_out['scores_all'], ".", markersize=4, color="royalblue", )
ax_reward.set_title(f" Reward (eval, greedy policy), mean={np.mean(eval_out['scores_all']):.2f}") 
ax_reward.set_xlabel("Eval episode")
ax_reward.set_ylabel("Reward")
ax_reward.grid(True)

final_fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save high-res
final_fig.savefig("final_metrics_6plots.png", dpi=300, bbox_inches="tight")
final_fig.savefig("final_metrics_6plots.pdf", bbox_inches="tight")
print("Saved: final_metrics_6plots.png and final_metrics_6plots.pdf")


env.close()
plt.ioff()
plt.show(block=True)

# fig.savefig("final_metrics.png", dpi=300, bbox_inches="tight")
# fig.savefig("final_metrics.pdf", bbox_inches="tight")

# %%
