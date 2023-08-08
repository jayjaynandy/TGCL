n_epochs=100
ckpt_no=25
tau=10
gnn_type='gin'
for ds in 'bbbp' 'clintox' 'bace' 'sider' 'tox21' 'toxcast' 'hiv' 'muv'
do
for seed in 0 1 2
do
    python finetune.py \
        --gnn_type $gnn_type \
        --JK last \
        --model_file ./ckpts/TGCL_tau$tau/$ckpt_no.pth \
        --runseed $seed \
        --dataset $ds \
        --lr 1e-3 \
        --lr_scale 1 \
        --emb_dim 300 \
        --decay 0 \
        --epochs $n_epochs \
        --result_dir results-$gnn_type-$tau-$ckpt_no-$n_epochs
done
done

