import sys
import torch
import os

from vis.SingleVisualizationModel import VisModel
from vis.projector import DVIProjector

from data.data_provider import NormalDataProvider
tar_CONTENT_PATH='/home/yifan/0ExpMinist/Default/02'
ref_CONTENT_PATH = '/home/yifan/00TrustVis/MINIST_ResNet18'
epoch = 5
sys.path.append(ref_CONTENT_PATH)
import Model.model as subject_model
net = eval("subject_model.{}()".format('resnet18'))

DEVICE = 'cuda:0'

ref_data_provider = NormalDataProvider(ref_CONTENT_PATH , net, epoch, epoch, 1, device='cuda:0', epoch_name='Epoch',classes=[],verbose=1)
tar_data_provider =  NormalDataProvider(tar_CONTENT_PATH , net, epoch, epoch, 1, device='cuda:0', epoch_name='Epoch',classes=[],verbose=1)

import time
save_dir = os.path.join(ref_data_provider.model_path, "Epoch_{}".format(epoch))
trans_model = torch.load(os.path.join(save_dir,'trans_model.m' )).to(DEVICE)

ENCODER_DIMS = [512,256,256,256,256,2]
DECODER_DIMS = [2,256,256,256,256,512]
model = VisModel(ENCODER_DIMS, DECODER_DIMS)


########################################################################################################################
#                                                      Evaluation                                                   #
########################################################################################################################
projector = DVIProjector(vis_model=model, content_path=ref_CONTENT_PATH, vis_model_name='contravis', device=DEVICE)

from vis.visualizer import visualizer
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
vis = visualizer(ref_data_provider, projector, 200, "tab10")
save_dir = os.path.join(ref_data_provider.content_path, "contraImg")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
# for i in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
vis.savefig(epoch, path=os.path.join(save_dir, "ref_{}.png".format(now)))
from vis.visualizer_for_tar import visualizer
vis = visualizer(ref_data_provider, tar_data_provider, trans_model, projector, 200,DEVICE, "tab10")
save_dir = os.path.join(ref_data_provider.content_path, "contraImg")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
vis.savefig(epoch, path=os.path.join(save_dir, "tar_{}.png".format(now)))


