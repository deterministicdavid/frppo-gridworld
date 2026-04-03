# Mirror Descent for Gridworld MDP

The purpose of the code in this repository is to test various mirror descent stepping schemes for solving tabular MDPs.

There is a single tabular MDP implemented in `GridworldMDP.py`. In principle the algorithms later just need access to size of state and action spaces and transition probabilities, rewards and the discount factor, so it would be easy to abstract this. 

## Algorithms implemented:
1. Policy iteration algorithm (PIA) in `PIA.py`
1. Policy iteration algorithm on softmax policies in `softmax_PIA.py`
1. Value iteration algorithm on softmax policies in `softmax_PIA.py`
1. Mirror descent algorithms in `mirror_descent.py`:
- vanilla explicit Euler stepping (which is precisely mirror descent)
- midpoint stepping (2nd order method)
- RK4 stepping (4th order method)

## Running this:
- To run the code, you will need to install [Python 3](https://www.python.org/downloads/), [NumPy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/). If you're using Poetry, you can just run `poetry install`.
- Run `main.py`. It's purpose is to provide plots showing impact of various chosen stepping schemes on how mirror descent performs.

## Results:
There are several experiments in `main.py` but currently it just runs those presented in the paper [A Fisher-Rao gradient flow for entropy-regularised Markov decision processes in Polish spaces](https://arxiv.org/abs/2310.02951)

