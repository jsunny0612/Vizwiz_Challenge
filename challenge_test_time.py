import logging
import os
import time
import random
import numpy as np
from datetime import datetime
from src.methods import *
from src.utils import create_submit_file, get_args, create_submit_file_for_new_idea
from src.utils.conf import cfg, load_cfg_fom_args, get_num_classes, get_domain_sequence
from src.models.load_model import load_model
from src.data.data import load_dataset
import json

logger = logging.getLogger(__name__)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def fix_seed(seed):
##################################################### func start
    deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
##################################################### func end

def evaluate(cfg):

    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    base_model = load_model(model_name=cfg.MODEL.ARCH, checkpoint_dir=os.path.join(cfg.CKPT_DIR, 'models'),
                            domain=cfg.CORRUPTION.SOURCE_DOMAIN, para_scale=cfg.MODEL.PARA_SCALE)
    base_model = base_model.cuda()

    if cfg.MODEL.ADAPTATION == 'parallel_psedo':
    ################################################################################################## if start
        model, param_names, param_names_ema = setup_parallel_psedo(base_model, cfg, num_classes)
    ################################################################################################## if end

    elif cfg.MODEL.ADAPTATION == 'parallel_psedo_contrast':
    ################################################################################################## elif start
        model, param_names, param_names_ema = setup_parallel_psedo_contrast(base_model, cfg, num_classes)
    ################################################################################################## elif end

    else:
    ################################################################################################## else start
        raise ValueError(f"Adaptation method '{cfg.MODEL.ADAPTATION}' is not supported!")
    ################################################################################################## else end

    ######################################################################################################### func start
    annotations = json.load(open( cfg.ANNOTATION_PATH))
    image_list = annotations["images"]
    indices_in_1k = [d['id'] for d in annotations['categories']]
    ######################################################################################################### func end

    # start evaluation
    testset, test_loader = load_dataset(cfg.CORRUPTION.DATASET, cfg.DATA_DIR,
                                        cfg.TEST.BATCH_SIZE,
                                        split='all',
                                        adaptation=cfg.MODEL.ADAPTATION,
                                        workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                        ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
                                        num_aug=cfg.TEST.N_AUGMENTATIONS,
                                        model_arch=cfg.MODEL.ARCH)

    for epoch in range(cfg.TEST.EPOCH):
    ############################################################################################################ for start
        if cfg.MODEL.ADAPTATION == 'parallel_psedo_contrast':
        ############################################################### if start
            results = create_submit_file_for_new_idea(model, data_loader=test_loader, mask = indices_in_1k, epoch = epoch, image_list = image_list)
        ############################################################### if end
        else:
        ############################################################### else start
            results = create_submit_file(model, data_loader=test_loader, mask = indices_in_1k)
        ############################################################### else end

        if ((epoch + 1) % 5) == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT, f'trained_{epoch+1}.pth'))

        file_path = os.path.join(cfg.OUTPUT, datetime.now().strftime(f'{epoch+1}epoch_prediction-%m-%d-%Y-%H:%M:%S.json'))
        with open(file_path, 'w') as outfile:
            json.dump(results, outfile)
        ############################################################################################################ for end

        if ((epoch + 1) % 5) == 0:
        ############################################ if start
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT,
                                    f'trained_{epoch+1}.pth'))
        ############################################ if end

        ############################################################### func start
        file_path = os.path.join(cfg.OUTPUT,
                                 datetime.now().strftime(f'{epoch+1}epoch_prediction-%m-%d-%Y-%H:%M:%S.json'))

        with open(file_path, 'w') as outfile:
            json.dump(results, outfile)
        ############################################################### func end

    ########################################################################################################################### for end

    return results


if __name__ == "__main__":
    args = get_args()
    args.output_dir = args.output_dir if args.output_dir else 'online_evaluation'
    load_cfg_fom_args(args.cfg, args.output_dir)
    logger.info(cfg)

    fix_seed(cfg.RNG_SEED)

    start_time = time.time()
    accs = []
    for domain in cfg.CORRUPTION.SOURCE_DOMAINS:
        logger.info("#" * 50 + f'evaluating domain {domain}' + "#" * 50)
        cfg.CORRUPTION.SOURCE_DOMAIN = domain
        results = evaluate(cfg)

    end_time = time.time()
    run_time = end_time - start_time
    hours = int(run_time / 3600)
    minutes = int((run_time - hours * 3600) / 60)
    seconds = int(run_time - hours * 3600 - minutes * 60)
    logger.info(f"total run time: {hours}h {minutes}m {seconds}s")
