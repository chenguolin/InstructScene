ROOM_TPYE=$1
TAG=$2
DEVICE=$3
EPOCH=$4
FVQVAE_TAG=$5
SG2SC_TAG=$6

# By default, $CFG is 1.
CFG=$7
if [ -z "$CFG" ]
then
  CFG=1.0
fi

# By default, $SG2SC_CFG is 1.
SG2SC_CFG=$8
if [ -z "$SG2SC_CFG" ]
then
  SG2SC_CFG=1.0
fi

CUDA_VISIBLE_DEVICES=$DEVICE \
python3 src/generate_sg.py configs/${ROOM_TPYE}_sg_diffusion_vq_objfeat.yaml \
  --tag $TAG --fvqvae_tag $FVQVAE_TAG --sg2sc_tag $SG2SC_TAG --checkpoint_epoch $EPOCH \
  --cfg_scale $CFG --sg2sc_cfg_scale $SG2SC_CFG \
  --n_scenes 0
# --n_scenes 5 --visualize --verbose --resolution 1024

# CUDA_VISIBLE_DEVICES=$DEVICE \
# python3 src/compute_fid_scores.py configs/${ROOM_TPYE}_sg_diffusion_vq_objfeat.yaml \
#   --tag $TAG --checkpoint_epoch $EPOCH

# CUDA_VISIBLE_DEVICES=$DEVICE \
# python3 src/synthetic_vs_real_classifier.py configs/${ROOM_TPYE}_sg_diffusion_vq_objfeat.yaml \
#   --tag $TAG --checkpoint_epoch $EPOCH
