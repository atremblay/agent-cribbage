# agent-cribbage

`python -m agent-cribbage Evaluate --number_games 100 --cuda --config_dir ./agent-cribbage/Evaluate/Train_agent --model_dir ./Train_agent`

`python -m agent-cribbage Evaluate --number_games 100 --cuda --config_dir ./agent-cribbage/Evaluate/Train_agent_Boltzmann --model_dir ./Train_agent_Boltzmann`

`python -m agent-cribbage Evaluate --number_games 100 --cuda --config_dir ./agent-cribbage/Evaluate/Train_EvalPlays --model_dir ./Train_EvalPlays`

`python -m agent-cribbage Evaluate --number_games 100 --cuda --config_dir ./agent-cribbage/Evaluate/Train_EvalCards --model_dir ./Train_EvalCards`


`python -m agent-cribbage Train --epochs 600 --number_games 100 --cuda --agent_yaml ./agent-cribbage/Configurations/Train_agent.yaml`

`python -m agent-cribbage Train --epochs 600 --number_games 100 --cuda --agent_yaml ./agent-cribbage/Configurations/Train_agent_Boltzmann.yaml`

`python -m agent-cribbage Train --epochs 600 --number_games 100 --cuda --agent_yaml ./agent-cribbage/Configurations/Train_EvalPlays.yaml`

`python -m agent-cribbage Train --epochs 600 --number_games 100 --cuda --agent_yaml ./agent-cribbage/Configurations/Train_EvalCards.yaml --dataepochs2keep 3`


`python -m agent-cribbage Play --agent_yaml ./agent-cribbage/Configurations/Play_agent_vs_human.yaml`

`python -m agent-cribbage Play --agent_yaml ./agent-cribbage/Configurations/Play_agent_vs_random.yaml --number_games 1000 --cuda`

`python -m agent-cribbage Play --agent_yaml ./agent-cribbage/Configurations/Play_agent_vs_EvalCards+EvalPlays.yaml --number_games 1000 --cuda`

`python -m agent-cribbage Play --agent_yaml ./agent-cribbage/Configurations/Play_EvalCards+EvalPlays_vs_random.yaml --number_games 1000 --cuda`

`python -m agent-cribbage Play --agent_yaml ./agent-cribbage/Configurations/Play_EvalCards_vs_random.yaml --number_games 1000 --cuda`
