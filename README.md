
# RL Q / Dyna-Q / RWM-Q Project

## Install
pip install gymnasium numpy

## Train
python main.py --mode train --algo q
python main.py --mode train --algo dyna-q

## Run
python main.py --mode run --algo q

## Structure
- rl_core: Q-learning (shared)
- models: environment models (tabular / world model)
- third_party: external model placeholder
