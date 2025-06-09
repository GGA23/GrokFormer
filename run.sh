python train.py --dataset cora --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6
python train.py --dataset citeseer --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 3 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.5 --feat_dropout 0.5 --prop_dropout 0.3
python train.py --dataset pubmed --lr 0.01 --weight_decay 5e-4 --nlayer 2 --k 2 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0
python train.py --dataset photo --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.4 --feat_dropout 0.4 --prop_dropout 0.4
python train.py --dataset physics --lr 0.01 --weight_decay 5e-3 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.3 --feat_dropout 0.3 --prop_dropout 0.7
python train.py --dataset wikics --lr 0.005 --weight_decay 5e-4 --dim 16 --hidden_dim 128 --k 2 --nlayer 1 --nheadss 1 --tran_dropout 0.6 --feat_dropout 0.4 --prop_dropout 0.1
python train.py --dataset penn --runs 5 --lr 0.01 --weight_decay 5e-5 --nlayer 1 --k 10 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.5 --feat_dropout 0.2 --prop_dropout 0.0
python train.py --dataset chameleon --lr 0.01 --weight_decay 5e-5 --nlayer 2 --k 5 --hidden_dim 128 --dim 32 --nheads 3 --tran_dropout 0.6 --feat_dropout 0.4 --prop_dropout 0.3
python train.py --dataset squirrel --lr 0.01 --weight_decay 5e-5 --nlayer 1 --k 6 --hidden_dim 128 --dim 32 --nheads 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.0
python train.py --dataset actor --lr 0.01 --weight_decay 5e-5 --nlayer 2 --k 2 --hidden_dim 128 --dim 32 --nheads 2 --tran_dropout 0.0 --feat_dropout 0.3 --prop_dropout 0.5
python train.py --dataset texas --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 1 --hidden_dim 128 --dim 16 --nheads 3 --tran_dropout 0.6 --feat_dropout 0.6 --prop_dropout 0.8
#python train.py --dataset texas --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 1 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.5 --feat_dropout 0.8 --prop_dropout 0.8
