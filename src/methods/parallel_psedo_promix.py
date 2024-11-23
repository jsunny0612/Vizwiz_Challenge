"""
Copyright to Tent Authors ICLR 2021 Spotlight
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
####################################
import torch.nn.functional as F
from src.utils.utils import *
from src.utils.fmix import *
######################################
from src.data.augmentations import get_tta_transforms
from src.utils.utils import deepcopy_model

from torch.autograd import Variable

# fmix = FMix()

class Parallel_psedo_promix(nn.Module):
######################################################################################################################## class start
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, ema_model, optimizer_teacher, mt_alpha, rst_m, ap, dataset_name, steps=1, episodic=False, num_aug=32, adaptation_type='otta', output_dir=None, use_memory=None, max_epoch = 10):
        super().__init__()

        device = torch.device("cuda")

        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

        self.model_ema = ema_model
        self.optimizer_teacher = optimizer_teacher

        self.model_state, self.optimizer_state, self.model_ema_state, self.optimizer_teacher_state, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer, self.model_ema, self.optimizer_teacher)

        self.transform = get_tta_transforms(dataset_name)
        self.softmax_entropy = softmax_entropy_cifar if "cifar" in dataset_name else softmax_entropy_imagenet
        self.num_aug = num_aug
        self.adaptation_type = adaptation_type
        self.output_dir = output_dir

        self.use_memory = use_memory
        self.psedo_lable_bank = torch.zeros((8900,max_epoch), device=device)

        ############################################## func start
        self.model_anchor.to(device)
        self.model_ema.to(device)
        ############################################### func end

######################################################################################################################## class end


    def forward(self, x, epoch, iter):
    ######################################################################################################################################## forward start

        if self.adaptation_type == 'ttba':
        ################################################################################################################ if for ttba start

            self.reset()
            self.model.train()

            if self.steps > 0:
                for _ in range(self.steps):
                    outputs = self.forward_and_adapt(x, self.model, self.optimizer)

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        ################################################################################################################ if for ttba end

        elif (self.adaptation_type == 'otta') or (self.adaptation_type == 'ttda'):
        ####################################################################################### if for otta, ttda start
            self.model.train()
            self.model_ema.train()

            if self.steps > 0:
                for _ in range(self.steps):
                    outputs = self.forward_and_adapt(x, self.optimizer, self.optimizer_teacher, epoch, iter)
        ######################################################################################## if for otta end

        return outputs
    ######################################################################################################################################### forward end


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, optimizer, optimizer_teacher, epoch, iter):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        # l = np.random.beta(4, 4)
        # l = max(l, 1 - l)
        #
        # all_inputs = x
        #
        # idx = torch.randperm(all_inputs.size(0))
        # input_a, input_b = all_inputs, all_inputs[idx]
        # mixed_input = l * input_a + (1 - l) * input_b
        # outputs = self. model(mixed_input)

        outputs = self.model(x)

        standard_ema = self.model_ema(x)
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]

        outputs_emas = []

        to_aug = anchor_prob.mean(0) < self.ap

        if to_aug:
            for i in range(self.num_aug):
                outputs_ = self.model_ema(self.transform(x)).detach()
                outputs_emas.append(outputs_)
        if to_aug:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema

        w = loss_weight(outputs_ema.detach())
        ########################################################################### new start
        # w_x = w.view(-1, 1).type(torch.FloatTensor)

        # with torch.no_grad():




        ########################################################################### new end

        # SCE loss
        loss_ce = self.softmax_entropy(outputs, outputs_ema.detach())
        loss_rce = self.softmax_entropy(outputs_ema.detach(), outputs)
        loss_sce = (0.1*loss_ce) + (1*loss_rce)

        if self.use_memory is not None:
        #################################################################################################### if start
            loss_sce = (w * loss_sce)
            mask = self.save_refine_psedo_lable(outputs_ema.detach(), epoch, iter)
            # mask = self.save_refine_psedo_lable(outputs_ema_fmix.detach(), epoch, iter)
            loss_sce = (mask * loss_sce).mean(0)
        #################################################################################################### if end
        else:
        ########################################################################################## else start
            loss_sce = (w * loss_sce).mean(0)
        ########################################################################################## else end

        loss_sce.backward()
        optimizer.step()
        optimizer.zero_grad()

        teacher_loss = softmax_entropy(outputs_ema).mean(0)
        teacher_loss.backward()
        optimizer_teacher.step()
        optimizer_teacher.zero_grad()

        with open(self.output_dir + f'/parallel_psedo_{str(self.model)[:11]}.txt', 'a') as f:
            f.write(f'loss: {loss_sce} teacher_loss: {teacher_loss} \n')

        return outputs

    def reset(self):

        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps


    def save_refine_psedo_lable(self, psedo, epoch, iter):
    ############################################################################## func start
        predictions = psedo.argmax(1)
        start = iter * len(psedo)
        end = start + len(psedo)
        mask = torch.ones(len(psedo), dtype=bool, device="cuda")

        self.psedo_lable_bank[start:end, epoch] = predictions



        if epoch == 0:
        ###################################### if start
            return mask
        ###################################### if end

        elif epoch < (self.use_memory + 1):
        ######################################################### else start
            select_past = range(epoch)

            for i in range(len(psedo)):
            ######################################## for start
                mask[i] = len(torch.where(self.psedo_lable_bank[start + i, select_past] != predictions[i])[0]) < 1
            ######################################## for end

        else:
        ######################################################### else start
            select_past = range(epoch - self.use_memory, epoch)

            for i in range(len(psedo)):
            ######################################## for start
                mask[i] = len(torch.where(self.psedo_lable_bank[start + i, select_past] != predictions[i])[0]) < 1
            ######################################## for end

        ######################################################### else end

        return mask
    ############################################################################## func end


    @staticmethod
    def configure_model(model):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        model.train()

        # disable grad, to (re-)enable only what tent updates
        model.requires_grad_(False)

        # configure norm for tent updates: enable grad + force batch statisics
        for nm, m in model.named_modules():
            if "parallel" in nm.split("."):
                m.requires_grad_(True)

        return model

    @staticmethod
    def configure_model_ema(model_ema):
    ######################################################################################################################## func start
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        model_ema.train()

        # disable grad, to (re-)enable only what tent updates
        model_ema.requires_grad_(False)

        for nm, m in model_ema.named_modules():
            if isinstance(m, (nn.LayerNorm)):
                m.requires_grad_(True)
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            elif isinstance(m, nn.GroupNorm):
                m.requires_grad_(True)
        return model_ema
    ######################################################################################################################## func end

    @staticmethod
    def collect_params(model):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """

        params = []
        names = []

        for nm, m in model.named_modules():
            if "parallel" in nm.split("."):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names


    @staticmethod
    def collect_params_ema(model):
    #################################################################################################################### func start
        params = []
        names = []

        for nm, m in model.named_modules():
            if isinstance(m, (nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
            if isinstance(m, nn.BatchNorm2d):
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            elif isinstance(m, nn.GroupNorm):
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names
    #################################################################################################################### func end


    @staticmethod
    def check_model(model):
        """Check model for compatability with tent."""
        is_training = model.training
        assert is_training, "tent needs train mode: call model.train()"
        param_grads = [p.requires_grad for p in model.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "tent needs params to update: " \
                               "check which require grad"
        assert not has_all_params, "tent should not update all params: " \
                                   "check which require grad"
        has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
        assert has_bn, "tent needs normalization for its optimization"


@torch.jit.script
def loss_weight(x: torch.Tensor) -> torch.Tensor:
################################################################################################ func start
    x = torch.nn.functional.softmax(x, dim=1)
    max_entropy = torch.log2(torch.tensor(1000))
    w = -torch.sum(x * torch.log2(x + 1e-5), dim=1)
    w = w / max_entropy
    w = torch.exp(-w)
    return w
################################################################################################ func end

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def mean_softmax_entropy(x:torch.Tensor)->torch.Tensor:
    temprature = 1
    x = x / temprature
    mean_probe_d=torch.mean(x.softmax(1),dim=0)
    entropy=-torch.sum(mean_probe_d*torch.log(mean_probe_d))
    return entropy


@torch.jit.script
def energy(x: torch.Tensor) -> torch.Tensor:
    """Energy calculation from logits."""
    temprature = 1
    x = -(temprature*torch.logsumexp(x / temprature, dim=1))
    if torch.rand(1) > 0.95:
        print(x.mean(0).item())
    return x


def copy_model_and_optimizer(model, optimizer, model_ema, optimizer_theacher):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())

    #################################################################################################################### func start
    model_ema_state = deepcopy(model_ema.state_dict())
    optimizer_theacher_state = deepcopy(optimizer_theacher.state_dict())

    model_anchor = deepcopy_model(model_ema)
    model_anchor.requires_grad_(False)

    #################################################################################################################### func end

    return model_state, optimizer_state, model_ema_state, optimizer_theacher_state, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

@torch.jit.script
def softmax_entropy_cifar(x, x_ema):  # -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def softmax_entropy_imagenet(x, x_ema):  # -> torch.Tensor:
    return -0.5 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)

