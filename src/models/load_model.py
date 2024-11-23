import os
import timm
import torch
from torchvision.models import resnet50, ResNet50_Weights, convnext_base, ConvNeXt_Base_Weights, efficientnet_b0, \
    EfficientNet_B0_Weights
from ..models import *
from copy import deepcopy
from src.models.convnext import ConvNeXt_para
from src.models.ResNet_para import ResNet_para
from src.models.vit_para import VisionTransformer_para
from src.models.base_model import BaseModel


def load_model(model_name, checkpoint_dir=None, domain=None, para_scale=0.1):
    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
    elif model_name == 'convnext_base':

        model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    elif model_name == 'resnet50-gn':
        ############################################################################################### elif start
        model = timm.create_model('resnet50_gn', pretrained=True)
    ############################################################################################### elif end

    elif model_name == 'resnet50_gn_para':
        #################################################################################################################### elif start
        model_args = dict(layers=[3, 4, 6, 3], norm_layer='groupnorm')
        model = ResNet_para(**dict(model_args))

        model_load = timm.create_model('resnet50_gn', pretrained=True)
        model_state = deepcopy(model_load.state_dict())

        model.load_state_dict(model_state, strict=False)
    #################################################################################################################### elif end

    elif model_name == 'vit_para':
        ######################################################################################################################### elif start
        model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        model = VisionTransformer_para(**dict(model_args))

        model_load = timm.create_model('vit_base_patch16_224', pretrained=True)
        model_state = deepcopy(model_load.state_dict())

        model.load_state_dict(model_state, strict=False)
    ########################################################################################################################### elif end

    elif model_name == 'convnext_base_para':
        ################################################################################################### elif start
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        model = ConvNeXt_para(3, 1000, **dict(model_args))

        model_load = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)
        model_state = deepcopy(model_load.state_dict())

        model.load_state_dict(model_state, strict=False)
        model = BaseModel(model, model_name)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.to(device)
    #################################################################################################### elif end

    elif model_name == 'convnext_base_para_384':
        ################################################################################################### elif start
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])
        model = ConvNeXt_para(3, 1000, **dict(model_args))

        model_load = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k_384', pretrained=True)
        model_state = deepcopy(model_load.state_dict())
        model.load_state_dict(model_state, strict=False)
        model = BaseModel(model, model_name)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.to(device)
    #################################################################################################### elif end

    elif model_name == 'convnextv2_huge_para':
        ######################################################################################## elif start
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_args = dict(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], use_grn=True, ls_init_value=None)
        model = ConvNeXt_para(3, 1000, **dict(model_args))

        model_load = timm.create_model('convnextv2_huge.fcmae_ft_in22k_in1k_384', pretrained=True)
        model_state = deepcopy(model_load.state_dict())

        model.load_state_dict(model_state, strict=False)
        model = BaseModel(model, model_name)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.to(device)
    ########################################################################################### elif end

    elif model_name == 'convnext_clip_para':
        ######################################################################################## elif start
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_args = dict(depths=[3, 4, 30, 3], dims=[384, 768, 1536, 3072], norm_eps=1e-5)
        model = ConvNeXt_para(3, 1000, **dict(model_args))

        model_load = timm.create_model('convnext_xxlarge.clip_laion2b_soup_ft_in1k', pretrained=True)
        model_state = deepcopy(model_load.state_dict())

        model.load_state_dict(model_state, strict=False)
        model = BaseModel(model, model_name)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.to(device)
    ########################################################################################### elif end

    else:
        raise ValueError('Unknown model name')

    return model