ROOM_TPYE=$1
TAG=$2
DEVICE=$3
FVQVAE_TAG=$4

CUDA_VISIBLE_DEVICES=$DEVICE \
python3 src/train_sg.py configs/${ROOM_TPYE}_sg_diffusion_vq_objfeat.yaml \
  --tag $TAG --fvqvae_tag $FVQVAE_TAG
