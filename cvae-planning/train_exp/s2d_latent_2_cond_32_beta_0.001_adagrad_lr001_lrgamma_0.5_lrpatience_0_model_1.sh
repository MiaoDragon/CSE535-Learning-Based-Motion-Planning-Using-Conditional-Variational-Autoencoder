cd ..
python train.py --model_dir /ilab/users/ym420/Documents/ylmiao/results/SMPNet/ --no_env 100 --no_motion_paths 4000 --val_ratio 0.1 \
--e_net_input_size 32 --e_net_output_size 28 --input_size 2 --latent_size 2 --cond_size 32 --beta 0.001 \
--model_id 1 --learning_rate 0.001 --lr_schedule_gamma 0.5 --lr_schedule_patience 0 \
--device 0 --num_epochs 100 --batch_size 256 \
--data_folder /freespace/local/ym420/course/cse535/data/ --start_epoch 0 --env_type s2d \
--world_size 20.0 20.0 --opt Adagrad
cd train_exp
#100x4000