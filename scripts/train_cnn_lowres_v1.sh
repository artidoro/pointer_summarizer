export PYTHONPATH=`pwd`
CUDA_VISIBLE_DEVICES=1 python training_ptr_gen/train.py \
    --save_path=cnn_v1_lowres_run2 \
    --reload_path=log/cnn_v1_lowres_run2/model/model_10000_1575425697 \
    --train_data_path=/remote/bones/user/public/vbalacha/cnn-dailymail/finished_files_cnn_lowres_v1/chunked/train_* \
    --eval_data_path=/remote/bones/user/public/vbalacha/cnn-dailymail/finished_files_cnn_lowres_v1/val.bin \
    --vocab_path=/remote/bones/user/public/vbalacha/cnn-dailymail/finished_files_cnn_lowres_v1/vocab \
    --lr_coverage 0.01 \
    --max_dec_steps 100 \
    --batch_size 8 >& log/cnn_v1_lowres_training_log &

