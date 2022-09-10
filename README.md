# CLEAN

## Usage
### 1 Preparation
generate 1x (2x, 4x, etc.) augmentation samples from the original sample using two open-source tools ([easyEDA](https://github.com/jasonwei20/eda_nlp), [nlpaug](https://github.com/makcedward/nlpaug)). You can also use our generated dataset, semrest_1(2, 4, 8).txt, semlaptop_1(2, 4, 8).txt, entity_1(2, 4) for 1x (2x, 4x, 8x) restaurant, laptop and CLIPEval datasets.
`python data_preprocess.py`
### 2 the first stage to obtain the accurate value of α
`python train_first_stage.py --aug_multi 2 --dataset 'semrest_2' --device 'cuda:0' --seed 1234 --batch_size 8 --accmulation_steps 2 --num_epoch_alpha 10 --lr 2e-5`
we freeze the bert model and only optimize the learning parameter of $\alpha$, and then get the alpha txt named '{dataset}-{num_epoch_alpha}.pt'
### 3 the second stage
`python train_second_stage.py --beta2 0.4 --aug_multi 1 --dataset 'semlaptop_1' --seed 3407 --batch_size 8 --accmulation_steps 2 --num_epoch 20 --num_epoch_alpha 10 --lr 2e-5 --l2reg 0.01`
`python train_second_stage.py --beta2 0.6 --aug_multi 2 --dataset 'semrest_2' --seed 1235 --batch_size 4 --accmulation_steps 4 --num_epoch 20 --num_epoch_alpha 10 --lr 2e-5 --l2reg 1e-5`
`python train_second_stage.py --task isa --model_name2 bert_spcno --beta2 0.3 --aug_multi 1 --dataset 'entity_1' --seed 1234 --batch_size 8 --accmulation_steps 2 --num_epoch 20 --num_epoch_alpha 10 --lr 2e-5 --l2reg 1e-5`
