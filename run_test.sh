#python3 test.py \
--baseroot './test_data/' \
--baseroot_mask './test_data_mask/' \
--results_path './results' \
--gan_type 'WGAN' \
--gpu_ids '1' \
--epoch 40 \
--batch_size 1 \
--num_workers 8 \
--pad_type 'zero' \
--activation 'elu' \
--norm 'none' \ 

#python test.py \
--baseroot './dgm_images/' \
--baseroot_mask './dgm_masks/' \
--results_path './results_dgm' \
--gan_type 'WGAN' \
--gpu_ids '6' \
--epoch 40 \
--batch_size 1 \
--num_workers 8 \
--pad_type 'zero' \
--activation 'elu' \
--norm 'none' \

python test.py \
--baseroot './dgm_images/' \
--baseroot_mask './dgm_masks/' \
--results_path './results_dgm' \
--gan_type 'WGAN' \
--gpu_ids '6' \
--epoch 40 \
--batch_size 1 \
--num_workers 8 \
--pad_type 'zero' \
--activation 'elu' \
--norm 'none' \