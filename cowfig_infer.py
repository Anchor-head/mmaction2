_base_ = ['configs\\recognition\\uniformerv2\\uniformerv2-large-p16-res336_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb.py']
load_from = 'ckpts\\uniformerv2-large-p16-res336_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb_20221219-9020986e.pth'

# model settings
model = dict(
    cls_head=dict(num_classes=33),
    data_preprocessor=dict(
        mean=[81.18819513, 84.80972548, 89.56644091],
        std=[52.37756894, 52.10623331, 51.52931835]))

dataset_type = 'VideoDataset'
data_root_val = 'data\\cows\\cowvids'
ann_file_test = 'data\\cows\\cowvids_list.txt'