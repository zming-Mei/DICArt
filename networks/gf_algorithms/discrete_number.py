import torch
from tqdm import tqdm
from datasets.dataloader import get_data_loaders_from_cfg, process_batch


def get_dataset_translation_min_max(train_loader, cfg):

    x_min, y_min, z_min = float('inf'), float('inf'), float('inf')
    x_max, y_max, z_max = float('-inf'), float('-inf'), float('-inf')

    for batch_sample in tqdm(train_loader, desc="Processing batches"):
        batch_sample = process_batch(
            batch_sample=batch_sample,
            device=cfg.device,
            pose_mode=cfg.pose_mode,
            PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS
        )

        trans_part = batch_sample['zero_mean_gt_pose'][:, -3:]

        x_min = min(x_min, trans_part[:, 0].min().item())
        y_min = min(y_min, trans_part[:, 1].min().item())
        z_min = min(z_min, trans_part[:, 2].min().item())

        x_max = max(x_max, trans_part[:, 0].max().item())
        y_max = max(y_max, trans_part[:, 1].max().item())
        z_max = max(z_max, trans_part[:, 2].max().item())

    return [x_min, x_max, y_min, y_max, z_min, z_max]


def translation_to_bins(trans_part, translation_status, bins_num):

    x_min, x_max = translation_status[0], translation_status[1]
    y_min, y_max = translation_status[2], translation_status[3]
    z_min, z_max = translation_status[4], translation_status[5]

    normalized_x = (trans_part[:, 0] - x_min) / (x_max - x_min + 1e-8)
    normalized_y = (trans_part[:, 1] - y_min) / (y_max - y_min + 1e-8)
    normalized_z = (trans_part[:, 2] - z_min) / (z_max - z_min + 1e-8)

    bins_x = torch.clamp((normalized_x * bins_num).long(), 0, bins_num - 1)
    bins_y = torch.clamp((normalized_y * bins_num).long(), 0, bins_num - 1)
    bins_z = torch.clamp((normalized_z * bins_num).long(), 0, bins_num - 1)

    bins = torch.stack([bins_x, bins_y, bins_z], dim=1)

    return bins


def bins_to_numbers(bins, translation_status, bins_num):

    x_min, x_max = translation_status[0], translation_status[1]
    y_min, y_max = translation_status[2], translation_status[3]
    z_min, z_max = translation_status[4], translation_status[5]

    normalized_x = bins[:, 0].float() / (bins_num - 1)
    normalized_y = bins[:, 1].float() / (bins_num - 1)
    normalized_z = bins[:, 2].float() / (bins_num - 1)

    numbers_x = normalized_x * (x_max - x_min) + x_min
    numbers_y = normalized_y * (y_max - y_min) + y_min
    numbers_z = normalized_z * (z_max - z_min) + z_min

    numbers = torch.stack([numbers_x, numbers_y, numbers_z], dim=1)

    return numbers