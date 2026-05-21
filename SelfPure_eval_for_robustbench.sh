CUDA_VISIBLE_DEVICES=1 python  ./SelfPure_eval_for_robustbench.py  \
    --data 'cifar10'\
    --desc 'Debenedetti2022Light_XCiT-M12' \
    --prefix 'com-CF10-apgd'\
    --batch-size 50 \
    --gattack 'linf-apgd' \
    --gattack-loss 'dlr'\
    --gattack-iter 100 \



#for enlarger test
# for i in {1..50}; do
#     echo "enlarge-$i"
#     # PERFIX = "enlarge-$i-CF10-cw"
#     CUDA_VISIBLE_DEVICES=2 /python  ./SelfPure_eval_for_robustbench.py  \
#     --data 'cifar10'\
#     --desc 'Debenedetti2022Light_XCiT-S12' \
#     --prefix "enlarge-$i-CF10-cw" \
#     --batch-size 50 \
#     --gattack 'linf-apgd' \
#     --gattack-eps $i/255 \
#     --gattack-loss 'ce'\
#     --gattack-iter 20 \

# done

