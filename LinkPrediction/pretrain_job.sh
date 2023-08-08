python pretrain_DSLA.py --dataset COLLAB --edge_pert_strength 0.001 --edit_learn --num_layer 5
python pretrain_DSLA.py --dataset IMDB-BINARY --edge_pert_strength 0.01 --edit_learn --num_layer 5
python pretrain_DSLA.py --dataset IMDB-MULTI --edge_pert_strength 0.01 --edit_learn --num_layer 5
python TGCL.py --dataset COLLAB --edge_pert_strength 0.001 --edit_learn --num_layer 5
python TGCL.py --dataset IMDB-BINARY --edge_pert_strength 0.01 --edit_learn --num_layer 5
python TGCL.py --dataset IMDB-MULTI --edge_pert_strength 0.01 --edit_learn --num_layer 5