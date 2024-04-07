# Replicate the Pythia models. They count one embedding layer in their table, but my num_layers doesn't count those; so always use their num_layers-1 for mine.
# 70M
python main_hlb_v030.py -s --savefile "../results_v030_pythia.csv" --seed 11000 --num_tries 1 --num_steps 10000 --num_epochs 2 --attn_type vanilla --use_x_norm 1 --use_qkv_norm 0 --use_qk_norm 0 1 --num_layers 5 --residual_depth 512 --num_heads 8 --batchsize 64
# 160M
python main_hlb_v030.py -s --append --savefile "../results_v030_pythia.csv" --seed 11000 --num_tries 1 --num_steps 10000 --num_epochs 2 --attn_type vanilla --use_x_norm 1 --use_qkv_norm 0 --use_qk_norm 0 1 --num_layers 11 --residual_depth 768 --num_heads 12 --batchsize 64
# 410M
python main_hlb_v030.py -s --append --savefile "../results_v030_pythia.csv" --seed 11000 --num_tries 1 --num_steps 10000 --num_epochs 2 --attn_type vanilla --use_x_norm 1 --use_qkv_norm 0 --use_qk_norm 0 1 --num_layers 21 --residual_depth 1024 --num_heads 16 --batchsize 64