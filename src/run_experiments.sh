python main_llm.py -s --num_tries 100 --num_steps 1000 --seed 1000 \\
--attn_type vanilla hydra --use_x_norm 0 1 --use_qkv_norm 0 1

python main_llm.py -s --append --num_tries 100 --num_steps 1000 --seed 1000 \\
--attn_type vanilla hydra --use_x_norm 0 --use_qk_norm 1

python main_llm.py -s --append --num_tries 100 --num_steps 1000 --seed 1000 \\
--attn_type vanilla hydra --use_x_norm 1 --use_qk_norm 1

mv ../results/results_llm.csv ../results/results_llm_1000_steps_100_tries.csv

python main_llm.py -s --num_tries 10 --num_steps 10000 --num_epochs 10 --seed 2000 \\
--attn_type vanilla hydra --use_x_norm 0 1 --use_qkv_norm 0 1

python main_llm.py -s --append --num_tries 10 --num_steps 10000 --num_epochs 10 --seed 2000 \\
--attn_type vanilla hydra --use_x_norm 0 --use_qk_norm 1

python main_llm.py -s --append --num_tries 10 --num_steps 10000 --num_epochs 10 --seed 2000 \\
--attn_type vanilla hydra --use_x_norm 1 --use_qk_norm 1

mv ../results/results_llm.csv ../results/results_llm_10_epochs_10_tries.csv