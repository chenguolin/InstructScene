ROOM_TPYE=$1
TAG=$2
DEVICE=$3
FVQVAE_TAG=$4

CUDA_VISIBLE_DEVICES=$DEVICE \
python3 src/train_sg2sc.py configs/${ROOM_TPYE}_sg2sc_diffusion_objfeat.yaml \
  --tag $TAG --fvqvae_tag $FVQVAE_TAG
