cd ..
python train.py --model_dir /ilab/users/ym420/Documents/ylmiao/results/SMPNet/ --no_env 100 --no_motion_paths 4000 \
--e_net_input_size 32 --e_net_output_size 28 --input_size 2 --latent_size 2 --cond_size 32 --beta 1.0 \
--model_id 0 --learning_rate 0.001 --device 3 --num_epochs 500 --batch_size 32 \
--data_folder /freespace/local/ym420/course/cse535/data/ --start_epoch 0 --env_type s2d \
--world_size 20.0 20.0 --opt Adam
cd train_exp
#100x4000