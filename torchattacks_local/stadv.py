import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms,datasets
from scipy import optimize

def flow_st(images,flows):
    images_shape = images.size()
    flows_shape = flows.size()
    batch_size = images_shape[0]
    H = images_shape[2]
    W = images_shape[3]
    basegrid = torch.stack(torch.meshgrid(torch.arange(0,H), torch.arange(0,W))) #(2,H,W)
    sampling_grid = basegrid.unsqueeze(0).type(torch.float32).cuda() + flows.cuda()
    sampling_grid_x = torch.clamp(sampling_grid[:,1],0.0,W-1.0).type(torch.float32)
    sampling_grid_y = torch.clamp(sampling_grid[:,0],0.0,H-1.0).type(torch.float32)

    x0 = torch.floor(sampling_grid_x).type(torch.int64)
    x1 = x0 + 1
    y0 = torch.floor(sampling_grid_y).type(torch.int64)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, W - 2)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 2)
    y1 = torch.clamp(y1, 0, H - 1)

    Ia = images[:,:,y0[0,:,:], x0[0,:,:]]
    Ib = images[:,:,y1[0,:,:], x0[0,:,:]]
    Ic = images[:,:,y0[0,:,:], x1[0,:,:]]
    Id = images[:,:,y1[0,:,:], x1[0,:,:]]

    x0 = x0.type(torch.float32)
    x1 = x1.type(torch.float32)
    y0 = y0.type(torch.float32)
    y1 = y1.type(torch.float32)

    wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
    wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
    wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
    wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)

    perturbed_image = wa.unsqueeze(0)*Ia+wb.unsqueeze(0)*Ib+wc.unsqueeze(0)*Ic+wd.unsqueeze(0)*Id

    return perturbed_image.type(torch.float32).cuda()

def flow_loss(flows,padding_mode='constant', epsilon=1e-8):
    paddings = (1,1,1,1)
    padded_flows = F.pad(flows,paddings,mode=padding_mode,value=0)
    shifted_flows = [
        padded_flows[:, :, 2:, 2:],  # bottom right (+1,+1)
        padded_flows[:, :, 2:, :-2],  # bottom left (+1,-1)
        padded_flows[:, :, :-2, 2:],  # top right (-1,+1)
        padded_flows[:, :, :-2, :-2]  # top left (-1,-1)
      ]
  #||\Delta u^{(p)} - \Delta u^{(q)}||_2^2 + # ||\Delta v^{(p)} - \Delta v^{(q)}||_2^2 
    loss=0
    for shifted_flow in shifted_flows:
        loss += torch.sum(torch.square(flows[:, 1] - shifted_flow[:, 1]) + torch.square(flows[:, 0] - shifted_flow[:, 0]) + epsilon).cuda()
    return loss.type(torch.float32)

def adv_loss(logits,targets,confidence=0.0):
    confidence=torch.tensor(confidence).cuda()
    real = torch.sum(logits*targets,-1)
    other = torch.max((1-targets)*logits-(targets*10000),-1)[0]
    return torch.max(other-real,confidence)[0].type(torch.float32)

def func(flows,input,target,model,const=0.05):
    # input = torch.from_numpy(input).cuda()
    # target = torch.from_numpy(target).cuda()
    flows = torch.from_numpy(flows).view((1,2,)+input.size()[2:]).cuda()
    flows.requires_grad=True
    pert_out = flow_st(input,flows)
    output = model(pert_out)
    L_flow = flow_loss(flows)
    L_adv = adv_loss(output,target)
    L_final = L_adv+const*L_flow
    model.zero_grad()
    L_final.backward()
    gradient = flows.grad.data.view(-1).detach()
    return L_final.item(),gradient

def StAdv(input,target,model):
    init_flows = np.zeros((1,2,)+input.size()[2:]).reshape(-1)
    results = optimize.fmin_l_bfgs_b(func,init_flows,args=(input,target,model))
    if 'CONVERGENCE' in results[2]['task']:
        flows = torch.from_numpy(results[0]).view((1,2,)+input.size()[2:])
        pert_out = flow_st(input,flows)
    else:
        print('none')
        return None
    return pert_out