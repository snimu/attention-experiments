# run on H100 -> don't double token capacity, just enjoy the extra speed!!!
# Without any norm, the grad_norm will be nan -> always use some norm
python main_hlb_v040.py -s --num_runs 10 --seed 1000 --use_x_norm 1 --use_qk_norm 0 1 --model_scale 0.5 1.0 5.0 10.0 --savefile "results_v040_1000_steps_10_tries_sqrt_dh.csv"
python main_hlb_v040.py -s --append --num_runs 10 --seed 1000 --use_x_norm 0 --use_qk_norm 1 --model_scale 0.5 1.0 5.0 10.0 --savefile "results_v040_1000_steps_10_tries_sqrt_dh.csv"