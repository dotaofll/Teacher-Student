Title         : Teacher-Student FrameWork for Zero-Shot Machine Translation
Author        : xxlucas
Logo          : True

[TITLE]

# Introduction 

TS-FrameWork is machine translation framework which is based on OpenNMT-py.

The main idea about how to implement is coming from **_A Teacher-Student Framework for
Zero-Resource Neural Machine Translation_**


Here is the paper :[arXiv:1705.00753]

And if you don't know what the zero-shot mean please Google it and you will figure it out.

# Usage

If you want to translate some low resource language to target language and the bi-lingual corpus are always not available.

So there is a commonly used method which is called pivot-based method.
It means that if the source-to-target parallel corpus are not available but we can find the source-to-pivot language pairs and the pivot-to-target language pairs.
Like this our problem becomes very easy to solve：train two NMT model separately for both source-to-pivot and pivot-to target language pairs.
But this also brings new problems that is two fold,on the one hand is training time is too long,on the other hand is error propagation that is the training error from the former model will be the input of the latter one.

Therefore,if we train a model once that all the issue will be vanished.

Assume that the source-to-pivot and pivot-to-target languages are all available.

**Step 0. preprocess the raw data **

``` python
mkdir data
cd data

mkdir teacher
mkdir student

python ../preprocess.py --train_src path_to_pivot --train_tgt path_to_target \
                  --save_data data/teacher
python ../preprocess.py --train_src path_to_source --train_tgt path_to_pivot \
                  --save_data data/student
```
**Step 1. train teacher model on pivot-to-target **

``` javascript
# the GNMT style teacher model
python train.py -data data/teacher -save_model ./model \
        -layers 8 -rnn_size 1024 -rnn_type GRU \
        -encoder_type brnn -decoder_type rnn \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 128 -batch_type sents -normalization sents  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 1e-3 \
        -max_grad_norm 5 -param_init 0  -param_init_glorot \
        -valid_steps 10000 -save_checkpoint_steps 10000 \
        -world_size 1 -gpu_ranks 0
```
**Step 2. train student model on source-to-pivot **

``` javascript
# the transformer student model
python train.py -data data/student/ -save_model student \
-layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
-encoder_type transformer -decoder_type transformer -position_encoding \
-train_steps 100000 -max_generator_batches 2 -dropout 0.1 \
-batch_size 4096 -batch_type tokens -normalization tokens -accum_count 2 \
-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
-label_smoothing 0.1 -save_checkpoint_steps 10000 \
-world_size 1 -gpu_ranks 0 --teacher_model_path model_step_200000.pt
```

You can evaluate the model with OpenNMT-py's api.
