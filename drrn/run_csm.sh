SEED=0

# Note: The 'snacktime' game has a different file extension (.z8) compared to other games (.z5).
GAME='snacktime'
LOG_FILE='csm_snacktime'

python train.py --output_dir ./logs/${LOG_FILE}_seed${SEED} \
        --rom_path ../games/${GAME}.z5 \ 
        --seed ${SEED} \
        --lm_path  ../download-models/gpt2 \
        --lm_ft_val_path ../download-models/jericho_walkthrough_data/walkthrough_${GAME}.json \
        --lm_k_max 30 \
        --lm_k_min 10 \
        --lm_k_thres 0.6 \
        --lm_k_start_at 20000\
        --lm_ft_buffer_min_start 20\
        --lm_ft_thres_type pos\
        --lm_ft_epoch 3\
        --lm_ft_batch_size 8\