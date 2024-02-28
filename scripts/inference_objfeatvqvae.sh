TAG=$1
DEVICE=$2
EPOCH=$3

CUDA_VISIBLE_DEVICES=$DEVICE \
python3 src/reconstruct_objfeatvqvae.py configs/threedfront_objfeat_vqvae.yaml \
  --tag $TAG --checkpoint_epoch $EPOCH
