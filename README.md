# FRPPO for Gridworld MDP

The purpose of the code in this repository is to test FR-PPO for solving tabular MDPs.

There is a single tabular MDP implemented in `GridworldMDP.py`. In principle the algorithms later just need access to size of state and action spaces and transition probabilities, rewards and the discount factor, so it would be easy to abstract this. 

## Algorithms implemented:
1. Policy iteration algorithm (PIA) in `PIA.py`
1. Policy iteration algorithm on softmax policies in `softmax_PIA.py`
1. Value iteration algorithm on softmax policies in `softmax_PIA.py`
1. FR-PPO descent algorithms in `fr_descent.py`:
1. Mirror Descent (KL penalty) in `mirror_descent.py` for comparison.

## Running this:
- To run the code, you will need to install [Python 3](https://www.python.org/downloads/), [NumPy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/). If you're using Poetry, you can just run `poetry install`.
- Run `main.py`. It's purpose is to provide plots showing impact of various chosen stepping schemes on how mirror descent performs.

## Results:
Run `main.py` to plot the figure linked below.

[Convergence plot](fr2_error_plot.pdf)