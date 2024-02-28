TAG=$1
DEVICE=$2

CUDA_VISIBLE_DEVICES=$DEVICE \
python3 src/train_objfeatvqvae.py configs/threedfront_objfeat_vqvae.yaml \
  --tag $TAG
