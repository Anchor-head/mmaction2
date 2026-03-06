from mmaction.apis import inference_recognizer, init_recognizer

# Initialize model
config_path = 'configs/recognition/uniformerv2/uniformerv2-large-p16-res336_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb.py'
checkpoint_path = 'ckpts/uniformerv2-large-p16-res336_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb_20221219-9020986e.pth'
model = init_recognizer(config_path, checkpoint_path)

cowvids = open('data/cowvids_list.txt', 'r').readlines()
for img in cowvids:
    img_path = f'/home/voinea/projects/rpp-banire/well-e/databases/LLaMARJO/appsheet-video/{img.strip()}.mp4'
    # test a single image
    result = inference_recognizer(model, img_path).pred_score
    label = inference_recognizer(model, img_path).pred_label
    # send the result to a json file
    print(label, result)
