
# RL Q / Dyna-Q / RWM-Q Project

## setup envirnment
module load python/3.10.10
module load python-venv

cd ~/workspace/ECE6756/V4
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
python main.py --mode train --algo rwm-predict

RWM debug plots are saved under `debug/` after training:
- `*_one_step_prediction_debug.png`
- `*_multi_step_prediction_debug.png`

## control how many episodes run
--episodes [number of episodes you want to run]

## Run
python main.py --mode run --algo q

## Structure
- rl_core: Q-learning (shared)
- models: environment models (tabular / world model)
- third_party: external model placeholder


## evaluation
python evaluation/sliding_window_plot.py --log-dir logs
python evaluation/sliding_window_plot.py --log-dir logs --tasks q rwm-q rwm-predict dyna-q-linear
python evaluation/sliding_window_plot.py --log-dir logs --window 200
python evaluation/sliding_window_plot.py --log-dir logs --output evaluation/my_plot.png
