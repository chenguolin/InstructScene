echo "Pickling 3D-Front dataset for faster data loading..."
python3 dataset/get_3dfront_pkl.py

echo "Pickling 3D-FUTURE dataset for faster data loading..."
python3 dataset/get_3dfuture_pkl.py bedroom
python3 dataset/get_3dfuture_pkl.py livingroom
python3 dataset/get_3dfuture_pkl.py diningroom
