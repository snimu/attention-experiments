# Run experiments for 1000 steps. This is definitely more than 1 epoch, but not much more.
#   1. 100 tries (to get really stable data)
#   2. approx 1 epoch (so that it doesnt take forever)
# The experiments:
#   1. Vanilla and Hydra attention each
#   2. Test every combination of x_norm and qkv_norm
#   3. Test every combination of x_norm and qk_norm (don't repeat tests without qk_norm)
python main_llm.py -s --num_tries 100 --num_steps 1000 --seed 1000 \\
--attn_type vanilla hydra --use_x_norm 0 1 --use_qkv_norm 0 1

python main_llm.py -s --append --num_tries 100 --num_steps 1000 --seed 1000 \\
--attn_type vanilla hydra --use_x_norm 0 --use_qk_norm 1

python main_llm.py -s --append --num_tries 100 --num_steps 1000 --seed 1000 \\
--attn_type vanilla hydra --use_x_norm 1 --use_qk_norm 1

mv ../results/results_llm.csv ../results/results_llm_1000_steps_100_tries.csv

# Run the same experiments, except
#   1. For 10 epochs (I don't know how many steps those are, so I just set it to 10000; it will stop earlier)
#   2. For 10 tries (10xepochs, 0.1xtries --> same time needs (approximately))
#   3. Different seed (to avoid duplication)
python main_llm.py -s --num_tries 10 --num_steps 10000 --num_epochs 10 --seed 2000 \\
--attn_type vanilla hydra --use_x_norm 0 1 --use_qkv_norm 0 1

python main_llm.py -s --append --num_tries 10 --num_steps 10000 --num_epochs 10 --seed 2000 \\
--attn_type vanilla hydra --use_x_norm 0 --use_qk_norm 1

python main_llm.py -s --append --num_tries 10 --num_steps 10000 --num_epochs 10 --seed 2000 \\
--attn_type vanilla hydra --use_x_norm 1 --use_qk_norm 1

mv ../results/results_llm.csv ../results/results_llm_10_epochs_10_tries.csv