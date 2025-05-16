CUDA_VISIBLE_DEVICES="0" python DICArt/runners/DICArt_eval.py \
--data_path ArtImage-High-level/ArtImage \
--sampling_steps 100 \
--batch_size 96 \
--seed 0 \
--cate_id 1 \
--saved_model_name Train11.17 \
--pts_encoder pointnet2 \
--pretrained_model_path_test   \
--eval 

