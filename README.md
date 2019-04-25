# agent-cribbage

There are 4 jobs that you can call in this module:

	1- Play against an agent or make play 2 agents agaisnt each other (requires a .yaml configuration)
	2- Preheat a model (Only works for The Show phase model).
	3- Train agents models (requires a .yaml configuration)
	4- Evaluate models in a given directory (requires a directory containing one or many configurations to evaluate, and the path where the models are located).


For you to play against our best agent:

`python -m agent-cribbage Play --agent_yaml ./agent-cribbage/Configurations/Play_agent_vs_human.yaml`


For you to play against the simple deterministic agent:

`python -m agent-cribbage Play --agent_yaml ./agent-cribbage/Configurations/Play_deterministic_vs_human.yaml`


To make play multiple agents against each other multiple games:

`python -m agent-cribbage Play --agent_yaml ./agent-cribbage/Configurations/Play_agent_vs_random.yaml --number_games 1000`


To preheat the model for phase 0 (The Show):

`python -m agent-cribbage Preheat --model conv --epochs 3`

To train a model giving a configuration:

`python -m agent-cribbage Train --epochs 500 --number_games 200 --agent_yaml ./agent-cribbage/Configurations/Train_agent_ConvLstm_Dealer+split_discarded_MC.yaml --dataepochs2keep 5`



To evaluate a directory containing multiple models:

`python -m agent-cribbage Evaluate --number_games 5000 --config_dir ./agent-cribbage/Evaluate/Train_agent_ConvLstm_Dealer+split_discarded --model_dir ./Train_agent_ConvLstm_Dealer+split_discarded_MC`



