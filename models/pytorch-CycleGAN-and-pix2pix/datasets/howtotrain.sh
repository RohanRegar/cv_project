python train.py --dataroot /DATA/soham/garments/preet/soumikrakshit/nyu-depth-v2/versions/1/nyu_data/data/nyu_pngs_from_mat --name mat4 --model pix2pix --direction AtoB --no_flip --dataset_mode template --input_nc 3 --output_nc 1 > output3.log 2>&1 &
python test.py --dataroot /DATA/soham/garments/preet/soumikrakshit/nyu-depth-v2/versions/1/nyu_data/data/nyu_pngs_from_mat --name mat5 --model pix2pix --direction AtoB --no_flip --dataset_mode template --input_nc 3 --output_nc 1