
CUDA_VISIBLE_DEVICES=7 python  ./smoothing_eval_for_flower102.py  \
    --desc 'flowwer102xcit' \
    --prefix 'com-flower102-square'\
    --batch-size 3 \
    --gattack 'linf-square' \
    --gattack-loss 'ce'\
    --gattack-iter 100 \
    --csvfile
    # --keep-original-x 