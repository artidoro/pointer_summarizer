export PYTHONPATH=`pwd`
CUDA_VISIBLE_DEVICES=1 python training_ptr_gen/train.py \
    --save_path=bbc_test_low_resource_pg \
    --train_data_path=../data/finished_files_wlabels_wner_wcoref_chains_reduced_1/chunked/train_* \
    --eval_data_path=../data/finished_files_wlabels_wner_wcoref_chains_reduced_1/val.bin \
    --vocab_path=../data/finished_files_wlabels_wner_wcoref_chains_reduced_1/vocab \
    --lr_coverage 0.15 \
    --max_dec_steps 20 \
    --batch_size 8 \


    #--reload_path=log/cnn_v1_lowres_run2/model/model_10000_1575425697 \
    #>& log/bbc_test_low_resource_pg_log &

