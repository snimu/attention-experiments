# run on H100 -> don't double token capacity, just enjoy the extra speed!!!
# Without any norm, the grad_norm will be nan -> always use some norm
# Different seed than on A10 to avoid redundancy
# More train than val epochs in hopes that run ends in validation; but if jumps in epochs are too high, don't waste too much time either
python main_hlb_v040.py -s --num_runs 10 --seed 2000 --linear 0 1 --use_x_norm 1 --use_qk_norm 0 1 --model_scale 0.5 1.0 2.0 --model_scale_method depth width both --num_epochs_train 2 --num_epochs_val 1 --savefile "results_v040_1000_steps_10_tries_sqrt_dh.csv"
python main_hlb_v040.py -s --append --num_runs 10 --seed 2000 --linear 0 1 --use_x_norm 0 --use_qk_norm 1 --model_scale 0.5 1.0 2.0 --model_scale_method depth width both --num_epochs_train 2 --num_epochs_val 1 --savefile "results_v040_1000_steps_10_tries_sqrt_dh.csv"