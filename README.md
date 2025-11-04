# Taxi-v3 Q-learning Project

This repository contains a Q-learning implementation for the OpenAI Gymnasium `Taxi-v3` environment. The project trains a tabular Q-learning agent to solve the Taxi pickup/dropoff task, visualizes training progress, and saves the learned Q-table and episode rewards for later playback.

## Files

- `project_1.ipynb` — Main Jupyter notebook. It:
  - Initializes the `Taxi-v3` environment (render_mode='rgb_array')
  - Implements Q-learning with an epsilon-greedy policy
  - Trains the agent for multiple episodes
  - Plots smoothed average reward across episodes
  - Saves `q_table_taxi.npy` and `rewards_eps_greedy_taxi.npy`
  - Demonstrates how to load the saved policy and render a short playback

- `q_table_taxi.npy` — Saved NumPy array of the learned Q-table (states × actions).
- `rewards_eps_greedy_taxi.npy` — Saved per-episode reward history for the training run.

## Project overview (short)

The Taxi-v3 task is a classic discrete RL environment: the agent navigates a grid to pick up and drop off a passenger at the requested location. Q-learning is used here as a tabular, off-policy algorithm that updates Q(s,a) values using the Bellman optimality update.

Key components in the notebook:
- update_q_value: performs the Q-learning update for (s, a, r, s')
- epsilon_greedy: picks random action with probability epsilon (exploration) otherwise argmax Q(s)
- Training loop: resets environment each episode, steps until termination or max steps, updates Q-table, decays epsilon
- Plotting: a smoothed moving-average of rewards (window=100) to observe learning
- Save/load: Q-table and reward history saved with `np.save` and reloaded with `np.load`

## Hyperparameters used in the notebook

These are the values present in the notebook. You can tune them to experiment.

- alpha (learning rate): 0.1
- gamma (discount factor): 0.9
- epsilon (start): 1.0
- min_epsilon: 0.01
- epsilon decay (per episode): multiplicative factor (example in notebook: epsilon*0.9995 or similar)
- episodes: 40000 (the notebook used a large number; reduce for quick tests)
- max_steps per episode: 500

## Dependencies

The notebook uses Python and the following packages (minimum versions suggested):

- Python 3.8+
- numpy
- gymnasium (the notebook uses `gymnasium` and the `Taxi-v3` environment)
- matplotlib
- imageio (used earlier for creating animations if desired)
- IPython (for Jupyter display helpers)

Install recommended packages in Windows PowerShell (example):

```powershell
python -m pip install --upgrade pip
pip install numpy gymnasium[box2d] matplotlib imageio ipython jupyter
```

Note: `gymnasium` installation may require extra system dependencies depending on your platform. If you get errors installing `gymnasium`, consult the package docs for platform-specific instructions.

## How to run (quick)

1. Open a PowerShell terminal in the project folder `c:\Users\fethi tech\Desktop\fifi\campus\datacamp\RL`.
2. (Optional) Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy gymnasium matplotlib imageio ipython jupyter
```

3. Start Jupyter and open `project_1.ipynb`:

```powershell
jupyter notebook
```

4. Re-run the notebook cells in order. Training may take long depending on `episodes` and your machine.

## How to quickly test the saved policy

If you already have `q_table_taxi.npy` in the folder, you can load it and run a short playback using the logic included near the end of `project_1.ipynb`:

- The notebook creates `policy_dict = {state: np.argmax(q_table[state])}`
- Reset environment with a fixed seed, then step for a small number of actions (e.g., `max_actions = 16`) using `policy_dict[state]` to choose actions
- Render frames with `env.render()` (the notebook uses `render_mode='rgb_array'` and collects frames)

Example snippet (already in the notebook):

```python
saved_policy = np.load("q_table_taxi.npy")
policy_dict = {i: np.argmax(saved_policy[i]) for i in range(saved_policy.shape[0])}
state, info = env.reset(seed=42)
for step in range(16):
    frame = env.render()
    action = policy_dict.get(state, env.action_space.sample())
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state
    if terminated or truncated:
        break
env.close()
```

## Reproducibility and seeding

The notebook sets seeds for reproducibility: `env.np_random, _ = seeding.np_random(42)`, `env.action_space.seed(42)`, and `np.random.seed(42)`. Results may still vary slightly with Gym/Gymnasium versions and rendering backends.

## Troubleshooting tips

- If `gymnasium` fails to import or a specific environment cannot be found, ensure you have a compatible version installed and the extras where required.
- If training is very slow, reduce `episodes` to something smaller (e.g., 1k or 5k) to test flow, then scale up.
- If rendering frames shows an error, check the `render_mode` and Gymnasium version; some render modes differ across versions.

## Suggested next steps / improvements

- Add command-line arguments or a small CLI to run training with configurable hyperparameters.
- Save checkpoints periodically (partial Q-table saves) to avoid long retraining when interrupted.
- Add a small unit test to verify Q-table shape and that saved files load correctly.
- Create a short animation (e.g., using `imageio`) or save rendered frames as GIF to visualize learned policy.
- Compare policies across different seeds and hyperparameter settings.

## Contact / Notes

This README was generated to summarize the Jupyter notebook `project_1.ipynb` in this folder. If you want, I can also:
- Add a script `run_train.py` to run training from the command line,
- Create a small playback script that writes a GIF of a rendered episode using the saved Q-table, or
- Reduce training episodes and re-run a quick training to produce sample plots and updated .npy files.

---

Path: `c:\Users\fethi tech\Desktop\fifi\campus\datacamp\RL\README.md`
