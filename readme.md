# Usage

Cover all the current file to TS/ folder.
And run ` ./train_baseline_stu.sh ` if process report a error about "there is no teacher model in xxx". Please
run `./train_baseline_tea.sh ` then run student model training script given above.

# Warning 

Untiling lanuch the training process you must make sure how many cuda device that avaluable.
If there are two gpus in the current node. There are some previous work you need to do.

First identify your current avaluable gpu id in your current node.
and if you want to use GPU id 1 and 3 of your OS, you will need to `export CUDA_VISIBLE_DEVICES=1,3`.
    
Second both `-world_size` and `-gpu_ranks` need to be set. E.g. `-world_size 4` `-gpu_ranks 0 1 2 3` will use 4 GPU on this node only. In this case we export avaluable gpu ids `1,3` so we modify `train_baseline_tea.sh` and `train_baseline_stu.sh` and set `-world_size 2` `-gpu_ranks 0 1` respectively.

## Please valide how many gpu avaluble in your current node and modify training script first
