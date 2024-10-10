import torch
from torch.autograd.functional import jvp
import torch.nn as nn
def mean_of_square_tensors(tensors):
    sum_ent=0
    sum_numel=0
    for tensor in tensors:
        sum_ent+=torch.sum(tensor**2)
        sum_numel+=tensor.numel()
    return sum_ent/sum_numel

def defense_noise(original_dy_dx,l2_norms_dy_dx,k,defense_type,layerwise=False):
    device=original_dy_dx[0].device
    if defense_type=='default':
        normalizer=[torch.ones_like(original_dy_dx[i]) for i in range(len(original_dy_dx))]
    elif defense_type=='ours':
        #Clamp to erase inf
        if layerwise:
            normalizer=[(torch.clamp(torch.sum(l2_norms_dy_dx[i]**2)**0.5/torch.clamp(torch.sum(original_dy_dx[i]**2)**0.5,min=1e-6),max=1e6))**0.5*torch.ones_like(original_dy_dx[i]) for i in range(len(original_dy_dx))]
        else:
            normalizer=[(torch.clamp(torch.abs(l2_norms_dy_dx[i]/torch.clamp(torch.abs(original_dy_dx[i]),min=1e-8)),max=1e6))**0.5 for i in range(len(original_dy_dx))]
    else:
        raise NotImplemented
    s=mean_of_square_tensors(normalizer)**0.5
    modified_dy_dx=[None]*len(original_dy_dx)
    for i in range(len(original_dy_dx)):
        noise=torch.randn(original_dy_dx[i].shape).to(device)
        normalize=normalizer[i]
        modified_dy_dx[i]=original_dy_dx[i]+noise*normalize*k/s
    return modified_dy_dx

def defense_dpsgd(original_dy_dx,l2_norms_dy_dx,k,defense_type,layerwise=False,clipping_threshold=0.1):
    device=original_dy_dx[0].device
    if defense_type=='default':
        normalizer=[torch.ones_like(original_dy_dx[i]) for i in range(len(original_dy_dx))]
    elif defense_type=='ours':
        #Clamp to erase inf
        if layerwise:
            normalizer=[(torch.clamp(torch.sum(l2_norms_dy_dx[i]**2)**0.5/torch.clamp(torch.sum(original_dy_dx[i]**2)**0.5,min=1e-6),max=1e6))**0.5*torch.ones_like(original_dy_dx[i]) for i in range(len(original_dy_dx))]
        else:
            normalizer=[(torch.clamp(torch.abs(l2_norms_dy_dx[i]/torch.clamp(torch.abs(original_dy_dx[i]),min=1e-6)),max=1e6))**0.5 for i in range(len(original_dy_dx))]
        normalizer=[torch.where(torch.abs(original_dy_dx[i]) > clipping_threshold, 0, normalizer[i]) for i in range(len(original_dy_dx))]
    else:
        raise NotImplemented
    s=mean_of_square_tensors(normalizer)**0.5
    # print([torch.max(s).item() for s in normalizer])
    modified_dy_dx=[None]*len(original_dy_dx)
    for i in range(len(original_dy_dx)):
        noise=torch.randn(original_dy_dx[i].shape).to(device)
        normalize=normalizer[i]
        modified_dy_dx[i]=torch.clamp(original_dy_dx[i],min=-clipping_threshold,max=clipping_threshold)+noise*normalize*k/s
    return modified_dy_dx

def defense_prune(original_dy_dx, l2_norms_dy_dx, prune_percentage, defense_type, layerwise=False): #Layerwise is for same format only
    
    if defense_type=='default':
        reference_dy_dx=[torch.abs(grad_tensor) for grad_tensor in original_dy_dx]
    elif defense_type=='ours':
        reference_dy_dx=[-(torch.clamp(torch.abs(l2_norms_dy_dx[i]/torch.clamp(torch.abs(original_dy_dx[i]),min=1e-8)),min=1e-6,max=1e6))**0.5 for i in range(len(original_dy_dx))]
        
    pruned_dy_dx = []

    flattened_ref_grad = torch.cat([ref_grad.view(-1) for ref_grad in reference_dy_dx])
    num_elements = flattened_ref_grad.numel()
    num_prune = int(num_elements * prune_percentage)
    sorted_grads, _ = torch.sort(flattened_ref_grad)
    threshold_value = sorted_grads[num_prune - 1]

    for orig_grad, ref_grad in zip(original_dy_dx, reference_dy_dx):
        pruned_grad = torch.where(ref_grad <= threshold_value, torch.zeros_like(orig_grad), orig_grad)
        pruned_dy_dx.append(pruned_grad)
        
    # print(sum([torch.sum(i==0) for i in pruned_dy_dx])/sum([torch.numel(i) for i in pruned_dy_dx]))
    
    return pruned_dy_dx

def compute_l2_norm_of_gradients(net, gt_data, gt_onehot_label, criterion):
    #This method provides the accurate gradient but is highly memory-expensive
    func_dy_dx = lambda x: torch.autograd.grad(criterion(net(x), gt_onehot_label), net.parameters(), create_graph=True)
    grad_dy_dx = torch.autograd.functional.jacobian(func_dy_dx,gt_data,vectorize=True,strategy='forward-mode')
    return [torch.norm(gra.view(*gra.shape[:gra.ndim-gt_data.ndim],-1),dim=-1) for gra in grad_dy_dx]

def compute_l2_norm_of_gradients_new(net, gt_data, gt_onehot_label, criterion, num_samples=10):
    """
    Compute the L2 norm of the gradients of the network parameters with respect to the input data.
    
    Parameters:
    - net: The neural network model.
    - gt_data: The input data.
    - gt_onehot_label: The ground truth one-hot labels.
    - criterion: The loss function.
    - num_samples: Number of random vector samples to use for the estimation.
    
    Returns:
    - A list containing the L2 norm of the gradients for each parameter.
    """
    #Note that this method is an approximation
    def func_dy_dx(x):
        loss = criterion(net(x), gt_onehot_label)
        grads = torch.autograd.grad(loss, net.parameters(), create_graph=True)
        return grads
    accumulated_sq_norms = [torch.zeros_like(param) for param in net.parameters()]

    for _ in range(num_samples):
        v = torch.randn_like(gt_data, requires_grad=True)
        _, grads = jvp(func_dy_dx, (gt_data,), (v,))
        for i, grad in enumerate(grads):
            accumulated_sq_norms[i] += grad.pow(2)
    
    gradient_norms = [torch.sqrt(accum_sq_norm / num_samples) for accum_sq_norm in accumulated_sq_norms]
    
    return gradient_norms

def defense_round(original_dy_dx,l2_norms_dy_dx,k,defense_type,layerwise=False):
    if defense_type=='default':
        normalizer=[torch.ones_like(original_dy_dx[i]) for i in range(len(original_dy_dx))]
    elif defense_type=='ours':
        if layerwise:
            normalizer=[(torch.clamp(torch.sum(l2_norms_dy_dx[i]**2)**0.5/torch.clamp(torch.sum(original_dy_dx[i]**2)**0.5,min=1e-6),min=1e-6,max=1e6))**0.5*torch.ones_like(original_dy_dx[i]) for i in range(len(original_dy_dx))]
        else:
            normalizer=[(torch.clamp(torch.abs(l2_norms_dy_dx[i]/torch.clamp(original_dy_dx[i],min=1e-6)),min=1e-6,max=1e6))**0.5 for i in range(len(original_dy_dx))]
    else:
        raise NotImplemented
        
    s=mean_of_square_tensors(normalizer)**0.5
    modified_dy_dx=[None]*len(original_dy_dx)
    for i in range(len(original_dy_dx)):
        normalize=normalizer[i]*k/s
        modified_dy_dx[i]=torch.round(original_dy_dx[i] / normalize) * normalize
    
    
    # for x in modified_dy_dx:
    #     print(torch.isnan(torch.sum(x)))
    return modified_dy_dx

def defended_gradients(net,gt_data,gt_label,criterion,noise_scale=0.1,defense='noise',defense_type='ours',num_samples=10,layerwise=False,clipping_threshold=0.1):
    #Net,input,target,criterion
    #Will directly use accurate if num_samples (#samples for sketching) is larger than 2**20
    #Warning: it is best to apply to a model without ReLU activations (Or other reasons to generate a gradient of 0)
    #If not then will use 1e-8 instead of 0
    #for pruning, k is ratio of pruning
    out = net(gt_data)
    y = criterion(out, gt_label)
    original_dy_dx = torch.autograd.grad(y, net.parameters())
    
    if defense_type=='ours':
        if num_samples>2**20:
            l2_norms_dy_dx = compute_l2_norm_of_gradients(net, gt_data, gt_label, criterion,num_samples=num_samples)
        else:
            l2_norms_dy_dx = compute_l2_norm_of_gradients_new(net, gt_data, gt_label, criterion,num_samples=num_samples)
    elif defense_type=='default':
        l2_norms_dy_dx=None
    else:
        raise NotImplemented

    if defense=='round':
        modified_dy_dx = defense_round(original_dy_dx, l2_norms_dy_dx, noise_scale, defense_type,layerwise)
    elif defense=='noise':
        modified_dy_dx = defense_noise(original_dy_dx, l2_norms_dy_dx, noise_scale, defense_type,layerwise)
    elif defense=='dpsgd':
        modified_dy_dx = defense_dpsgd(original_dy_dx, l2_norms_dy_dx, noise_scale, defense_type,layerwise,clipping_threshold=clipping_threshold)
    elif defense=='prune':
        modified_dy_dx = defense_prune(original_dy_dx, l2_norms_dy_dx, noise_scale, defense_type,layerwise)
    else:
        raise NotImplementedError
    return modified_dy_dx

def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def mean_of_tensors(tensors):
    sum_ent=0
    sum_numel=0
    for tensor in tensors:
        sum_ent+=torch.sum(tensor)
        sum_numel+=tensor.numel()
    return sum_ent/sum_numel

def mean_of_square_tensors(tensors):
    sum_ent=0
    sum_numel=0
    for tensor in tensors:
        sum_ent+=torch.sum(tensor**2)
        sum_numel+=tensor.numel()
    return sum_ent/sum_numel

def clone_model(model):
    cloned_model = type(model)().to(device)
    cloned_model.load_state_dict(model.state_dict())
    return cloned_model

class ReLU2LeakyReLU(nn.Module):
    def __init__(self, resnet_model, negative_slope=0.01):
        super().__init__()
        
        # Copy the layers of the original ResNet
        self.resnet = resnet_model
        
        # Set the negative_slope for LeakyReLU
        self.negative_slope = negative_slope
        
        # Replace all ReLU activations with LeakyReLU
        self._replace_relu_with_leakyrelu()

    def _replace_relu_with_leakyrelu(self):
        # Recursively search through all layers and replace ReLU with LeakyReLU
        for name, module in self.resnet.named_children():
            if isinstance(module, nn.ReLU):
                setattr(self.resnet, name, nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True))
            elif len(list(module.children())) > 0:
                self._replace_relu_with_leakyrelu_recursive(module)

    def _replace_relu_with_leakyrelu_recursive(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True))
            elif len(list(child.children())) > 0:
                self._replace_relu_with_leakyrelu_recursive(child)

    def forward(self, x):
        return self.resnet(x)
