python3 main.py --output-dir 'output/test/' --dataset-list-root 'dataset_list' --batch-size 16

python3 main.py --output-dir 'output/ep100bs64lr0.01/' --dataset-list-root 'dataset_list' --batch-size 64 --epochs 100 --lr 0.01



python3 main.py --output-dir 'output/img_size448_ep100bs64lr0.001/' \
 --dataset-list-root 'dataset_list' --batch-size 64 --epochs 100 --lr 0.001 --image-size 448 



python3 main.py --output-dir 'output/img_size448_ep100bs32lr0.0003/' \
 --dataset-list-root 'dataset_list' --batch-size 32 --epochs 100 --lr 0.0003 --image-size 448 
