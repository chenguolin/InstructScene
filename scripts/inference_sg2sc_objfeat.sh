ROOM_TPYE=$1
TAG=$2
DEVICE=$3
EPOCH=$4
FVQVAE_TAG=$5

# By default, $CFG is 1.
CFG=$6
if [ -z "$CFG" ]
then
  CFG=1.0
fi

CUDA_VISIBLE_DEVICES=$DEVICE \
python3 src/generate_sg2sc.py configs/${ROOM_TPYE}_sg2sc_diffusion_objfeat.yaml \
  --tag $TAG --fvqvae_tag $FVQVAE_TAG --checkpoint_epoch $EPOCH \
  --cfg_scale $CFG \
  --n_scenes 0
# --n_scenes 5 --visualize --verbose --resolution 1024

# CUDA_VISIBLE_DEVICES=$DEVICE \
# python3 src/compute_fid_scores.py configs/${ROOM_TPYE}_sg2sc_diffusion_objfeat.yaml \
#   --tag $TAG --checkpoint_epoch $EPOCH

# CUDA_VISIBLE_DEVICES=$DEVICE \
# python3 src/synthetic_vs_real_classifier.py configs/${ROOM_TPYE}_sg2sc_diffusion_objfeat.yaml \
#   --tag $TAG --checkpoint_epoch $EPOCH
