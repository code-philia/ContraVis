import sys
import torch
import os

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

from data.edge_dataset import DVIDataHandler
from vis.trainer import DVITrainer
from vis.custom_weighted_random_sampler import CustomWeightedRandomSampler
from vis.SingleVisualizationModel import VisModel
from umap.umap_ import find_ab_params
import numpy as np

from vis.projector import DVIProjector
from vis.spatial_edge_constructor import AlignedSpitalEdgeConstructor
from vis.losses import UmapLoss, ReconstructionLoss, DVILoss, SingleVisLoss, DummyTemporalLoss
import time
from Transformation.transformModel import TransformModelTrainer

from data.data_provider import NormalDataProvider
tar_CONTENT_PATH='/home/yifan/0ExpMinist/Default/02'
ref_CONTENT_PATH = '/home/yifan/00TrustVis/MINIST_ResNet18'

sys.path.append(ref_CONTENT_PATH)
import Model.model as subject_model
net = eval("subject_model.{}()".format('resnet18'))

epoch = 1
S_N_EPOCHS = 5
DEVICE = 'cuda:0'

ref_data_provider = NormalDataProvider(ref_CONTENT_PATH , net, epoch, epoch, 1, device='cuda:0', epoch_name='Epoch',classes=[],verbose=1)
tar_data_provider =  NormalDataProvider(tar_CONTENT_PATH , net, epoch, epoch, 1, device='cuda:0', epoch_name='Epoch',classes=[],verbose=1)

ref_data = ref_data_provider.train_representation(epoch)
ref_data = ref_data.reshape(ref_data.shape[0],ref_data.shape[1])
tar_data = tar_data_provider.train_representation(epoch)
tar_data = tar_data.reshape(tar_data.shape[0],tar_data.shape[1])


Transformation_Trainer = TransformModelTrainer(ref_data,tar_data,ref_data_provider,tar_data_provider,epoch,epoch,'cuda:0',15)
trans_model,tar_data_mapped,tar_proxy_mapped,ref_reconstructed  = Transformation_Trainer.transformation_train(num_epochs=2000)

save_dir = os.path.join(ref_data_provider.model_path, "Epoch_{}".format(epoch))
##### save the transformation model
torch.save(trans_model, os.path.join(save_dir,'trans_model.m' ))
t0 = time.time()
spatial_cons = AlignedSpitalEdgeConstructor(data_provider=ref_data_provider,epoch=epoch, s_n_epochs=S_N_EPOCHS,b_n_epochs=0,n_neighbors=15,transed_tar=tar_data_mapped,tar_provider=tar_data_provider)


edge_to, edge_from, probs, feature_vectors, attention = spatial_cons.construct()
t1 = time.time()

probs = probs / (probs.max()+1e-3)
eliminate_zeros = probs> 1e-3    #1e-3
edge_to = edge_to[eliminate_zeros]
edge_from = edge_from[eliminate_zeros]
probs = probs[eliminate_zeros]

dataset = DVIDataHandler(edge_to, edge_from, feature_vectors, attention)

n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
# chose sampler based on the number of dataset
if len(edge_to) > pow(2,24):
    sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
else:
    sampler = WeightedRandomSampler(probs, n_samples, replacement=True)

edge_loader = DataLoader(dataset, batch_size=2000, sampler=sampler, num_workers=8, prefetch_factor=10)

# ########################################################################################################################
# #                                                       TRAIN                                                          #
# ########################################################################################################################
# Define visualization models
ENCODER_DIMS = [512,256,256,256,256,2]
DECODER_DIMS = [2,256,256,256,256,512]
LAMBDA1 = 1
PATIENT = 5
MAX_EPOCH =20
VIS_MODEL_NAME = 'contravis'
model = VisModel(ENCODER_DIMS, DECODER_DIMS)
# Define Losses
negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)
umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
single_loss_fn = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA1)
temporal_loss_fn = DummyTemporalLoss(DEVICE)
criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=LAMBDA1, lambd2=0.0,device=DEVICE)
# Define training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)

trainer = DVITrainer(model, criterion, optimizer, lr_scheduler, edge_loader=edge_loader, DEVICE=DEVICE)

t2=time.time()
trainer.train(PATIENT, MAX_EPOCH)
t3 = time.time()
# save result
save_dir = ref_data_provider.model_path
trainer.record_time(save_dir, "time_{}".format(VIS_MODEL_NAME), "complex_construction", str(epoch), t1-t0)
trainer.record_time(save_dir, "time_{}".format(VIS_MODEL_NAME), "training", str(epoch), t3-t2)
save_dir = os.path.join(ref_data_provider.model_path, "Epoch_{}".format(epoch))
 ##### save the visulization model
trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))

########################################################################################################################
#                                                      VISUALIZATION                                                   #
########################################################################################################################
projector = DVIProjector(vis_model=model, content_path=ref_CONTENT_PATH, vis_model_name='contravis', device=DEVICE)
emb = projector.batch_project(epoch, ref_data)
print("emb",emb)
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


