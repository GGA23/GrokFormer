python train.py --dataset cora --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6
python train.py --dataset citeseer --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 3 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.5 --feat_dropout 0.5 --prop_dropout 0.3
python train.py --dataset pubmed --lr 0.01 --weight_decay 5e-4 --nlayer 2 --k 2 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0
python train.py --dataset photo --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.4 --feat_dropout 0.4 --prop_dropout 0.4
python train.py --dataset physics --lr 0.01 --weight_decay 5e-3 --nlayer 1 --k 2 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.3 --feat_dropout 0.3 --prop_dropout 0.7
python train.py --dataset wikics --lr 0.005 --weight_decay 5e-4 --dim 16 --hidden_dim 128 --k 2 --nlayer 1 --nheads 1 --tran_dropout 0.6 --feat_dropout 0.4 --prop_dropout 0.1
python train.py --dataset penn --runs 5 --lr 0.01 --weight_decay 5e-5 --nlayer 1 --k 10 --hidden_dim 128 --dim 16 --nheads 1 --tran_dropout 0.5 --feat_dropout 0.2 --prop_dropout 0.0
python train.py --dataset chameleon --lr 0.01 --weight_decay 5e-5 --nlayer 2 --k 5 --hidden_dim 128 --dim 32 --nheads 3 --tran_dropout 0.6 --feat_dropout 0.4 --prop_dropout 0.3
python train.py --dataset squirrel --lr 0.01 --weight_decay 5e-5 --nlayer 1 --k 6 --hidden_dim 128 --dim 32 --nheads 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.0
python train.py --dataset actor --lr 0.01 --weight_decay 5e-5 --nlayer 2 --k 2 --hidden_dim 128 --dim 32 --nheads 2 --tran_dropout 0.0 --feat_dropout 0.3 --prop_dropout 0.5
python train.py --dataset texas --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 1 --hidden_dim 128 --dim 16 --nheads 3 --tran_dropout 0.6 --feat_dropout 0.6 --prop_dropout 0.8
#python train.py --dataset texas --lr 0.01 --weight_decay 5e-4 --nlayer 1 --k 1 --hidden_dim 128 --dim 16 --nheads 2 --tran_dropout 0.5 --feat_dropout 0.8 --prop_dropout 0.8



#parameter selection ranges we used for the Cora dataset: --lr 0.01; --weight_decay 5e-4; --tran_dropout 0.7 or 0.8; --feat_dropout 0.5 or 0.6; --prop_dropout 0.6 or 0.7; --nlayer 1 or 2; --k 1 or 2; --dim 16, 32, or 64; --nheads 1~5; norm:none.
#for citeseer: --lr 0.01; --weight_decay 5e-4; --tran_dropout 0.5 or 0.6; --feat_dropout 0.5~0.7; --prop_dropout 0.3~0.6; --nlayer 1 or 2; --k 1~3; --dim 16; --nheads 1~4; norm:none.
#for pubmed: --lr 0.01; --weight_decay 5e-4; --tran_dropout 0.3 or 0.4; --feat_dropout 0.2~0.4; --prop_dropout 0.0; --nlayer 1 or 2; --k 2 or 3; --dim 16 or 32; --nheads 1; norm:none.
#for photo: --lr 0.01; --weight_decay 5e-4; --tran_dropout 0.4; --feat_dropout 0.3 or 0.4; --prop_dropout 0.1~0.4; --nlayer 1; --k 2 or 3; --dim 16; --nheads 1 or 2; norm:none.
#for physics: --lr 0.01; --weight_decay 5e-3; --tran_dropout 0.2 or 0.3; --feat_dropout 0.5 or 0.3; --prop_dropout 0.6 or 0.7; --nlayer 1; --k 1~3; --dim 16 or 32; --nheads 1; norm:none.
#for penn: --lr 0.01; --weight_decay 5e-5; --tran_dropout 0.2 ~ 0.5; --feat_dropout 0.2~0.4; --prop_dropout 0.0 ~ 0.2; --nlayer 1 or 2; --k 10; --dim 16 or 32; --nheads 1 or 2; norm:none.
#for chameleon: --lr 0.01; --weight_decay 5e-5; --tran_dropout 0.3 ~ 0.7; --feat_dropout 0.4 or 0.5; --prop_dropout 0.1 ~ 0.3; --nlayer 2; --k 5; --dim 32; --nheads 2~5; norm:none.
#for squirrel: --lr 0.01; --weight_decay 5e-5; --tran_dropout 0.4~0.8; --feat_dropout 0.4 or 0.5; --prop_dropout 0.0 or 0.1; --nlayer 1 or 2; --k 6; --dim 32; --nheads 1 or 2; norm:none.
#for actor: --lr 0.01; --weight_decay 5e-5; --tran_dropout 0.0 ~ 0.2; --feat_dropout 0.3 ~ 0.5; --prop_dropout 0.4~0.7; --nlayer 1 or 2; --k 2; --dim 16 or 32; --nheads 2 or 3; norm:none.
#for texas: --lr 0.01; --weight_decay 5e-4; --tran_dropout 0.5 ~ 0.8; --feat_dropout 0.6 ~ 0.8; --prop_dropout 0.7 or 0.8; --nlayer 1 or 2; --k 1 or 2; --dim 16 or 32; --nheads 1~4; norm:none.
