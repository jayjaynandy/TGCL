TAU=6
for ds in 'COLLAB' 'IMDB-BINARY' 'IMDB-MULTI'
do
for ckpt in 100
do
for seed in 0 1 2
do
    python finetune_edgepred_hptune.py --num_layer 5 --dataset $ds --model_file ckpts/$ds/TGCL_tau$TAU/$ckpt.pth --seed $seed
done
done
done
