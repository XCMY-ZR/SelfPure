CUDA_VISIBLE_DEVICES=2 python  ./eval_target_flower102.py  \
    --data-dir '$your_data$' \
    --log-dir '$your_Log$' \
    --desc 'flower102xcit' \
    --prefix 'comxcit-flower102-pgd'\
    --batch-size 3 \
    --gattack 'linf-pgd' \
    --gattack-loss 'cw'\
    --gattack-iter 40 \
    --sattack-target \



CUDA_VISIBLE_DEVICES=2 python  ./eval_target_flower102.py  \
    --data-dir '$your_data$' \
    --log-dir '$your_Log$' \
    --desc 'flower102xcit' \
    --prefix 'comxcit-flower102-pgd'\
    --batch-size 3 \
    --gattack 'linf-pgd' \
    --gattack-loss 'cw'\
    --gattack-iter 40 \
    --sattack-HN \
    --sattack-target \
