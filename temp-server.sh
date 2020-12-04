export ALFRED_ROOT=$(pwd); 

python models/train/train_seq2seq.py \
  --data data/json_feat_2.1.0 \
  --model seq2seq_im_mask \
  --dout exp/model:{model},name:pm_and_subgoals_01 \
  --splits data/splits/oct21.json \
  --gpu \
  --batch 8 \
  --pm_aux_loss_wt 0.1 \
  --subgoal_aux_loss_wt 0.1 \
  --preprocess

####################################
# PREPROCESSING
####################################
# preprocess language part of data
python models/train/train_seq2seq.py \
  --data data/json_feat_2.1.0 \
  --splits data/splits/oct21.json \
  --preprocess

# setup smaller train/val/test splits for faster training
python -m ipdb -c continue data/filter.py \
  --out-splits data/splits/kitchens-full.json \
  --train-scenes $(seq 1 30) \
  --test-scenes $(seq 1 30)

python -m ipdb -c continue data/filter.py \
  --out-splits data/splits/kitchens-half.json \
  --train-scenes $(seq 15 30) \
  --test-scenes $(seq 15 30)

python -m ipdb -c continue data/filter.py \
  --out-splits data/splits/kitchens-small.json \
  --train-scenes 24 \
  --test-scenes 24