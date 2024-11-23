import os
import pickle
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from .Dataset_Idx import Dataset_Idx
from .data_list import *
from .selectedRotateImageFolder import SelectedRotateImageFolder


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_transform(dataset_name, adaptation, num_augment=1, model_arch = None):
######################################################################################## func start
    """
    Get transformation pipeline
    :param dataset_name: Name of the dataset
    :param adaptation: Name of the adaptation method
    :return: transforms
    """

    if model_arch == 'convnextv2_huge_para':
    ##################################################### if start
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            normalize
        ])
    ##################################################### if end

    elif model_arch == 'convnext_base_para_384':
    ########################################################## elif start
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            normalize
        ])
    ########################################################## elif end

    elif model_arch == 'convnext_base_para':
    ################################################################ elif start
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    ############################################################### elif end

    elif model_arch == 'convnext_clip_para':
    ################################################################ elif start
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize
        ])
    ############################################################### elif end

    else:
    ##################################################### elif start
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    ##################################################### elif end

######################################################################################## func end
    return transform

def load_challenge_data(root, transforms, batch_size=64, workers=4, ckpt=None):
######################################################################################################################## func start
    assert os.path.exists(root), f'Path {root} does not exist'

    validdir = os.path.join(root)
    teset = SelectedRotateImageFolder(validdir, transforms, original=False,
                                      rotation=False)

    ckpt_dir = os.path.join(ckpt, 'challenge')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, 'list.pickle')
    if not os.path.exists(ckpt_path):
        idx = torch.randperm(len(teset))
        idx = [i.item() for i in idx]
        with open(ckpt_path, 'wb') as f:
            pickle.dump(idx, f)
    else:
        with open(ckpt_path, 'rb') as f:
            idx = pickle.load(f)
    teset.samples = [teset.samples[i] for i in idx]
    teset.switch_mode(True, False)
    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle=False,
                                           num_workers=workers, pin_memory=True)
######################################################################################################################## func end
    return teset, teloader

def load_dataset(dataset, root, batch_size=64, workers=4, split='train', adaptation=None, ckpt=None, num_aug=1, transforms=None, model_arch = None):
    transforms = get_transform(dataset, adaptation, num_aug, model_arch) if transforms is None else transforms

    if dataset == 'challenge':
    ####################################################################################################################
        return load_challenge_data(root=os.path.join(root, 'challenge'), batch_size=batch_size, workers=workers,
                               transforms=transforms, ckpt=ckpt)
    ####################################################################################################################
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def load_dataset_idx(dataset, root, batch_size=64, workers=4, split='train', adaptation=None, ckpt=None, num_aug=1):
    dataset, _ = load_dataset(dataset, root, batch_size, workers, split, adaptation, ckpt, num_aug)
    dataset_idx = Dataset_Idx(dataset)
    data_loader = torch.utils.data.DataLoader(dataset_idx, batch_size=batch_size, shuffle=False, num_workers=workers,
                                              drop_last=False)
    return dataset_idx, data_loader
