# botorch: https://botorch.readthedocs.io/en/stable/models.html#botorch.models.gp_regression.SingleTaskGP

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from Flappybird import FlappyBirdGame

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize

def tprint(*args, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}]", *args, **kwargs)

# Settings Seeds    
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.use_deterministic_algorithms(True)

# fixes params
input_dimension = 5
output_dimension = 2
epsilon_min = 0.001
batch_size = 512
buffer_size = 750000
steps_per_gradient_update = 10
max_episode_step = 8000

def state_to_input(state):
    bird_y, bird_vel, bird_dist, pipe_top, pipe_btm = state

    max_bird_y = 768
    max_dist = 400
    max_gap_height = 768

    bird_y_normalized = (2 * bird_y / max_bird_y) - 1
    bird_vel_normalized = bird_vel / 10
    bird_dist_normalized = (2 * bird_dist / max_dist) - 1
    pipe_top_normal = (2 * pipe_top / max_gap_height) - 1
    pipe_btm_normal = (2 * pipe_btm / max_gap_height) - 1

    return torch.hstack((
        torch.tensor([bird_y_normalized], dtype=torch.float32),
        torch.tensor([bird_vel_normalized], dtype=torch.float32),
        torch.tensor([bird_dist_normalized], dtype=torch.float32),
        torch.tensor([pipe_top_normal], dtype=torch.float32),
        torch.tensor([pipe_btm_normal], dtype=torch.float32)
    ))

def seed_all(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available(): 
        torch.mps.manual_seed(seed)
        
def greedy_eval_mean_pipes(q_net, n_eval_games: int, eval_seed: int) -> float:
    seed_all(eval_seed)
    eval_env = FlappyBirdGame()

    pipes = []
    q_net.eval()
    with torch.no_grad():
        for _ in range(n_eval_games):
            done = False
            obs = state_to_input(eval_env.reset_game())
            ep_pipe = 0
            ep_steps = 0

            while (not done) and (ep_steps < max_episode_step):
                action = int(torch.argmax(q_net(obs)).item())  # greedy
                obs_next, reward, done, score_game_pipe = eval_env.step(action)
                obs = state_to_input(obs_next)
                ep_pipe = score_game_pipe
                ep_steps += 1

            pipes.append(ep_pipe)

    eval_env.close()
    q_net.train()
    return float(np.mean(pipes))

def run_training(
    learning_rate: float,
    epsilon_reduction_factor: float,
    gamma: float,
    hidden_dimension: int,
    n_games: int = 1000,       #number of episodes per iteration
    n_eval_games: int = 30,    #eval episodes after each trial for BO to optimize 
    trial_seed: int = 42,      #seed to ensure 
):
    seed_all(trial_seed)

    q_net = torch.nn.Sequential(
        torch.nn.Linear(input_dimension, hidden_dimension),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dimension, hidden_dimension),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dimension, output_dimension),
    )

    optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    # load env 
    env = FlappyBirdGame()
    actions = np.arange(2)

    # Buffers 
    obs_buffer = torch.zeros((buffer_size, input_dimension))
    obs_next_buffer = torch.zeros((buffer_size, input_dimension))
    action_buffer = torch.zeros(buffer_size).long()
    reward_buffer = torch.zeros(buffer_size)
    done_buffer = torch.zeros(buffer_size)

    # Logging
    episode_rewards = []
    pipe_scores = []

    # Training loop
    epsilon = 1.0
    step_count = 0

    for i in range(n_games):
        score = 0.0
        episode_pipe_score = 0
        episode_step = 0
        done = False

        env_observation = env.reset_game()
        observation = state_to_input(env_observation)

        epsilon = (epsilon - epsilon_min) * epsilon_reduction_factor + epsilon_min

        while (not done) and (episode_step < max_episode_step):
            if np.random.rand() < epsilon:
                action = int(np.random.choice(actions))
            else:
                with torch.no_grad():
                    action = int(torch.argmax(q_net(observation)).item())

            env_observation_next, reward, done, score_game_pipe = env.step(action)
            observation_next = state_to_input(env_observation_next)

            score += float(reward)
            episode_pipe_score = score_game_pipe

            # store in replay
            buffer_index = step_count % buffer_size
            obs_buffer[buffer_index] = observation
            obs_next_buffer[buffer_index] = observation_next
            action_buffer[buffer_index] = action
            reward_buffer[buffer_index] = float(reward)
            done_buffer[buffer_index] = float(done)

            observation = observation_next

            # Learn
            if step_count > batch_size and step_count % steps_per_gradient_update == 0:
                batch_idx = np.random.choice(
                    np.minimum(buffer_size, step_count),
                    size=batch_size,
                    replace=False
                )

                out = q_net(obs_buffer[batch_idx])
                val = out[np.arange(batch_size), action_buffer[batch_idx]]

                with torch.no_grad():
                    out_next = q_net(obs_next_buffer[batch_idx])
                    target = reward_buffer[batch_idx] + \
                             gamma * torch.max(out_next, dim=1).values * (1 - done_buffer[batch_idx])

                loss = loss_function(val, target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=20)
                optimizer.step()

            episode_step += 1
            step_count += 1

        pipe_scores.append(episode_pipe_score)
        episode_rewards.append(score)

    env.close()

    # evaluation episods across 2 seeds, when mean
    eval_seed1 = trial_seed + 12345
    eval_seed2 = trial_seed + 22345

    mean1 = greedy_eval_mean_pipes(q_net, n_eval_games=n_eval_games, eval_seed=eval_seed1)
    mean2 = greedy_eval_mean_pipes(q_net, n_eval_games=n_eval_games, eval_seed=eval_seed2)

    obj = 0.5 * (mean1 + mean2)

    # Optional debug prints (training)
    K = min(50, len(pipe_scores))
    tprint(
        f"    train_last{K}_pipes={np.mean(pipe_scores[-K:]):.3f} | "
        f"train_last{K}_reward={np.mean(episode_rewards[-K:]):.3f}"
    )
    tprint(f"    eval_pipes_seed1={mean1:.3f} | eval_pipes_seed2={mean2:.3f} | obj={obj:.3f}")

    return obj
 


# BoTorch: map x in [0,1]^d -> hyperparams 
def decode_params(x01: torch.Tensor):
    """
    x01: tensor shape (d,) in [0,1]
    return: (learning_rate, eps_red, gamma, hidden_dim)
    """
    x = x01.detach().cpu().numpy()

    # lr log-uniform [1e-5, 3e-3]
    lr_min, lr_max = 1e-5, 3e-3
    lr = 10 ** (np.log10(lr_min) + x[0] * (np.log10(lr_max) - np.log10(lr_min)))

    # epsilon reduction factor [0.95, 0.99999]
    eps_min, eps_max = 0.95, 0.99999
    eps_red = eps_min + x[1] * (eps_max - eps_min)

    # gamma [0.90, 0.9995]
    g_min, g_max = 0.90, 0.9995
    gamma = g_min + x[2] * (g_max - g_min)

    # hidden_dim in {128, 256, 512}
    choices = np.array([128, 256, 512])
    idx = int(np.round(x[3] * (len(choices) - 1)))
    hidden_dim = int(choices[idx])

    return lr, eps_red, gamma, hidden_dim

# Botorch optimization loop
def botorch_optimize(
    n_init: int = 8,
    n_iter: int = 15,
    n_games_trial: int = 200, 
    n_eval_games: int = 50,
    trial_seed: int = 42,
):
    device = torch.device("cpu")
    dtype = torch.double

    d = 4  # lr, eps_red, gamma, hidden_dim_index
    bounds = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)]).to(device)

    # initial random points
    X = torch.rand(n_init, d, device=device, dtype=dtype)
    Y_vals = []

    tprint("\n--- Initial random trials ---")
    for i in range(n_init):
        lr, eps_red, gamma, hidden_dim = decode_params(X[i])
        y = run_training(lr, eps_red, gamma, hidden_dim, n_games=n_games_trial, n_eval_games=n_eval_games, trial_seed=trial_seed)
        Y_vals.append([y])
        tprint(f"[init {i}] y={y:.3f} | lr={lr:.2e}, eps_red={eps_red:.6f}, gamma={gamma:.4f}, hidden={hidden_dim}")

    Y = torch.tensor(Y_vals, device=device, dtype=dtype)

    tprint("\n--- BO iterations ---")
    for t in range(n_iter):
        # Fit GP on standardized Y
        model = SingleTaskGP(X, standardize(Y))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        #LogExpectedImprovement acqf
        best_f = standardize(Y).max().item()
        from botorch.acquisition.analytic import LogExpectedImprovement
        acqf = LogExpectedImprovement(model=model, best_f=best_f)

        X_next, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )

        lr, eps_red, gamma, hidden_dim = decode_params(X_next[0])
        y_next = run_training(lr, eps_red, gamma, hidden_dim, n_games=n_games_trial,n_eval_games=n_eval_games, trial_seed=trial_seed)

        X = torch.cat([X, X_next], dim=0)
        Y = torch.cat([Y, torch.tensor([[y_next]], device=device, dtype=dtype)], dim=0)

        best_so_far = Y.max().item()
        tprint(f"[iter {t}] y={y_next:.3f} (best={best_so_far:.3f}) | lr={lr:.2e}, eps_red={eps_red:.6f}, gamma={gamma:.4f}, hidden={hidden_dim}")

    best_idx = int(torch.argmax(Y).item())
    best_lr, best_eps, best_gamma, best_hidden = decode_params(X[best_idx])

    tprint("\n=== BEST FOUND ===")
    print(f"best_y = {Y[best_idx].item():.3f}")
    print(f"learning_rate={best_lr:.6e}")
    print(f"epsilon_reduction_factor={best_eps:.6f}")
    print(f"gamma={best_gamma:.4f}")
    print(f"hidden_dimension={best_hidden}")

    return (best_lr, best_eps, best_gamma, best_hidden), X, Y


if __name__ == "__main__":
    best_params, X_hist, Y_hist = botorch_optimize(
        n_init=10,
        n_iter=20,
        n_games_trial=3000,  
        n_eval_games=50,
        trial_seed=42,
    )