
# RL Q / Dyna-Q / RWM-Q Project

## setup envirnment
module load python/3.10.10
module load python-venv

cd ~/workspace/ECE6756/V3
python -m venv .venv
source .venv/bin/activate
pip install numpy gymnasium


## ever time open a new terminal
source .venv/bin/activate

## Train
python main.py --mode train --algo q
python main.py --mode train --algo dyna-q-discret
python main.py --mode train --algo dyna-q-linear
python main.py --mode train --algo rwm-q

## control how many episodes run
--episodes [number of episodes you want to run]

## Run
python main.py --mode run --algo q

## Structure
- rl_core: Q-learning (shared)
- models: environment models (tabular / world model)
- third_party: external model placeholder

