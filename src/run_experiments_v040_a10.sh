# run on A10 -> half token capacity
# Without any norm, the grad_norm will be nan -> always use some norm
# More train than val epochs in hopes that run ends in validation; but if jumps in epochs are too high, don't waste too much time either
python main_hlb_v040.py -s --num_runs 10 --seed 1000 --linear 0 1 --use_x_norm 1 --use_qk_norm 0 1 --model_scale 0.5 1.0 --token_capacity_factor 0.4  --num_epochs_train 3 --num_epochs_val 2 --savefile "results_v040_preliminary_1000_steps_10_tries_sqrt_dh.csv"
python main_hlb_v040.py -s --append --num_runs 10 --seed 1000 --linear 0 1 --use_x_norm 0 --use_qk_norm 1 --model_scale 0.5 1.0 --token_capacity_factor 0.4 --num_epochs_train 3 --num_epochs_val 2 --savefile "results_v040_preliminary_1000_steps_10_tries_sqrt_dh.csv"