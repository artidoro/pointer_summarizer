export PYTHONPATH=`pwd`
#MODEL=$1
python training_ptr_gen/decode.py \
    --save_path=dm_v1_lowres_run2 \
    --reload_path=log/dm_v1_lowres_run2/model/model_40000_1575440338 \
    --decode_data_path=/remote/bones/user/public/vbalacha/cnn-dailymail/finished_files_dm_lowres_v1/test.bin \
    --vocab_path=/remote/bones/user/public/vbalacha/cnn-dailymail/finished_files_dm_lowres_v1/vocab \
    --max_dec_steps 100 \
    --beam_size 3  >& log/cnn_decode_log &

