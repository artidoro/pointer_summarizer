export PYTHONPATH=`pwd`
CUDA_VISIBLE_DEVICES=2 python training_ptr_gen/train.py \
    --save_path=dm_v1_lowres_run2 \
    --reload_path=log/dm_v1_lowres_run2/model/model_15000_1575426717 \
    --train_data_path=/remote/bones/user/public/vbalacha/cnn-dailymail/finished_files_dm_lowres_v1/chunked/train_* \
    --eval_data_path=/remote/bones/user/public/vbalacha/cnn-dailymail/finished_files_dm_lowres_v1/val.bin \
    --vocab_path=/remote/bones/user/public/vbalacha/cnn-dailymail/finished_files_dm_lowres_v1/vocab \
    --lr_coverage 0.01 \
    --max_dec_steps 100 \
    --batch_size 8 >& log/dm_v1_lowres_training_log &

