# Navigation
## Project Details
In this project, we train an agent to navigate a large, square world to
collect yellow bananas while avoiding blue bananas. We use a
[Unity ML-Agents](https://unity.com/products/machine-learning-agents)
based environment provided by Udacity for this exercise. More details
about the environment is provided below.

#### State space
The state space has 37 dimensions and contains the agent's velocity,
along with ray-based perception of objects around the agent's
forward direction. Given this information, the agent has to learn
how to best select actions.

#### Action space
The agent can take one of four discrete actions, corresponding to:

| action_id | action        |
|-----------|---------------|
| 0         | move forward  |
| 1         | move backward |
| 2         | turn left     |
| 3         | turn right    |

#### Reward
A reward of +1 is provided for collecting a yellow banana, and a reward
of -1 is provided for collecting a blue banana. The goal of the agent is
to collect as many yellow bananas as possible while avoiding blue bananas.

#### Definition of solved
The task is episodic - there are 300 time-steps in each episode (by default;
this setting can be changed). The environment is considered solved when the
agent gets an average score of +13 over 100 consecutive episodes.

## Getting Started

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__:
	```bash
	conda create --name drlnd python=3.6
	conda activate drlnd
	```
	- __Windows__:
	```bash
	conda create --name drlnd python=3.6
	conda activate drlnd
	```

2. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
    ```bash
    git clone https://github.com/nsriram13/rl-navigation.git
    cd rl-navigation/python
    pip install .
    ```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

4. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

5. This repository uses pre-commit hooks for auto-formatting and linting.
    * Run `pre-commit install` to set up the git hook scripts - this installs flake8 formatting, black
    auto-formatting as pre-commit hooks.
    * Run `gitlint install-hook` to install the gitlint commit-msg hook
    * (optional) If you want to manually run all pre-commit hooks on a repository,
    run `pre-commit run --all-files`. To run individual hooks use `pre-commit run <hook_id>`.

## Instructions
Follow the instructions in `Navigation.ipynb` to train the agent.
