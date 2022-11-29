from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf

from modules.evaluations import get_val_pair, get_val_data, perform_val, test_registration
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm


flags.DEFINE_string('cfg_path', './configs/arc_res50.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_folder_path', 'v3_dataset/v3', 'path to image pairs') #LYJ
flags.DEFINE_string('img_path', '', 'path to input image')
# flags.DEFINE_string('img_path', 'test_img/test/test_0.png', 'path to input image')
flags.DEFINE_string('registered_img_path', 'test_img/test/test_420.png', 'path to registered image') #LYJ


def main(_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()

    if FLAGS.registered_img_path: #LYJ
        print("[*] Encode {} to ./output_embeds.npy".format(FLAGS.registered_img_path))
        reg_img = cv2.imread(os.path.join(cfg['test_dataset'], FLAGS.registered_img_path))
        reg_img = cv2.resize(reg_img, (cfg['input_size'], cfg['input_size']))
        reg_img = reg_img.astype(np.float32) / 255.
        reg_img = cv2.cvtColor(reg_img, cv2.COLOR_BGR2RGB)
        if len(reg_img.shape) == 3:
            reg_img = np.expand_dims(reg_img, 0)
        reg_embeds = l2_norm(model(reg_img))

        np.save('./{}_output_embeds.npy'.format(os.path.join(cfg['test_dataset'], FLAGS.registered_img_path.replace('.png', ''))), reg_embeds)
        print('[*] Image {} registered'.format(FLAGS.registered_img_path)) #LYJ
    
    if FLAGS.img_path:
        print("[*] Encode {} to ./output_embeds.npy".format(FLAGS.img_path))
        img = cv2.imread(os.path.join(cfg['test_dataset'], FLAGS.img_path))
        img = cv2.resize(img, (cfg['input_size'], cfg['input_size']))
        img = img.astype(np.float32) / 255.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        embeds = l2_norm(model(img))

        # np.save('./output_embeds.npy', embeds)
        print(embeds.shape)
        #LYJ
        pred_result, dist = test_registration(reg_embeds, embeds, threshold=1.34) # LFW 1.34 / AgeDB30 1.43 / CFP-FP 1.53
        print('[*] Test img {} is registered: {} (dist: {})'.format(FLAGS.img_path, pred_result, dist))
    elif FLAGS.img_folder_path: #LYJ
        print('[*] Loading custom dataset...')
        arr, issame = get_val_pair(cfg['test_dataset'], FLAGS.img_folder_path, binary=False)
        print("[*] Perform Evaluation on custom set...")
        print(arr.shape, issame)
        acc_test, best_th = perform_val(
            cfg['embd_shape'], batch_size=cfg['batch_size'], 
            model=model, carray=arr, issame=issame, nrof_folds=10
        )
        print("    acc {:.4f}, th: {:.2f}".format(acc_test, best_th))
    else:
        print("[*] Loading LFW, AgeDB30 and CFP-FP...")
        lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = \
            get_val_data(cfg['test_dataset'])

        print("[*] Perform Evaluation on LFW...")
        acc_lfw, best_th = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, lfw, lfw_issame,
            is_ccrop=cfg['is_ccrop'])
        print("    acc {:.4f}, th: {:.2f}".format(acc_lfw, best_th))

        print("[*] Perform Evaluation on AgeDB30...")
        acc_agedb30, best_th = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, agedb_30,
            agedb_30_issame, is_ccrop=cfg['is_ccrop'])
        print("    acc {:.4f}, th: {:.2f}".format(acc_agedb30, best_th))

        print("[*] Perform Evaluation on CFP-FP...")
        acc_cfp_fp, best_th = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, cfp_fp, cfp_fp_issame,
            is_ccrop=cfg['is_ccrop'])
        print("    acc {:.4f}, th: {:.2f}".format(acc_cfp_fp, best_th))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
