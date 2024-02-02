## install
Follow the installation guides for stable-retro here:
https://github.com/Farama-Foundation/stable-retro

Afterwards move the F-zero integration directory to your stable-retro install.

## play script
ex. python3 play.py --model_path models/fzero-PPO-mute-city.zip --model_type PPO --state start-mute-city-1
arguments:
*	--model_path (string) – Path to model
*	--model_type (string) – Type of model
*	--game (string) – Game that the agent plays
*	--state (string) – State for the agent.
*	--n_env (int) - Number of env’s.
*	--record(bool) – If you want to record the agent to a BK2

## train script
ex. python3 train.py --model_type PPO --model_policy CnnPolicy --state start-mute-city-1
arguments:
*	--model_path (string) – Path to an existing model.
*	--model_type (string) – Type of model
*	--model_policy (string) – Het type policy dat het model gebruikt bv. CnnPolicy.
*	--game (string) – Game that the agent plays
*	--state (string) – State for the agent
*	--n_env (int) - Number of env’s.
*	--tensorboard_log (string) – path for tensor logs
