# Code for "Fusformer: A Transformer-based Fusion Approach for Hyperspectral Image Super-resolution"
# Author: Jin-Fan Hu

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import DatasetFromHdf5
# from model import PanNet,summaries
from model_Ours_5_block import *
import numpy as np
import torch.nn.functional as F
import scipy.io as sio
import shutil
from torch.utils.tensorboard import SummaryWriter
import time
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam, ssim, scc
import os
from cal_ssim import SSIM, set_random_seed



# ================== Pre-Define =================== #
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = True  ###????????
cudnn.deterministic = True
# cudnn.benchmark = False
# ============= HYPER PARAMS(Pre-Defined) ==========#
# lr = 0.001
# epochs = 1010
# ckpt = 50
# batch_size = 32
# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)   # optimizer 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

lr = 2e-4
epochs = 300
ckpt_step = 1
batch_size = 16

def cosine_similarity_loss(z):
    """
    计算端元一致性损失（基于余弦相似度）
    输入:
        z: (M, d_model) 端元特征
    输出:
        loss: 标量，一致性损失
    """
    M = z.size(0)
    norm = torch.norm(z, p=2, dim=1, keepdim=True)  # (M, 1)
    z_norm = z / norm  # 归一化
    
    # 计算余弦相似度矩阵
    sim_matrix = torch.mm(z_norm, z_norm.t())  # (M, M)
    
    # 排除对角线元素
    sim_matrix = sim_matrix - torch.eye(M).to(z.device)
    
    # 计算平均负相似度
    return -sim_matrix.sum() / (M * (M - 1))

def loss(reconstructed, gt):
    """
    计算光谱角距离损失
    :param reconstructed: 重建的高光谱图像，形状为 [B, N, H, W]
    :param gt: 真实的高光谱图像，形状为 [B, N, H, W]
    :return: 光谱角距离损失
    """
    dot_product = torch.sum(reconstructed * gt, dim=1)
    norm_reconstructed = torch.norm(reconstructed, dim=1)
    norm_gt = torch.norm(gt, dim=1)
    cos_angle = dot_product / (norm_reconstructed * norm_gt + 1e-8)
    # 避免余弦值超出 [-1, 1] 范围
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)
    return torch.mean(angle)+PLoss(reconstructed, gt)

model = HSI_ReNet_g(128, 4, 20).cuda()
# model = nn.DataParallel(model)
# model_path = "Weights/.pth"
# if os.path.isfile(model_path):
#     # Load the pretrained Encoder
#     model.load_state_dict(torch.load(model_path))
#     print('Network is Successfully Loaded from %s' % (model_path))
# from torchstat import stat
# stat(model, input_size=[(31, 16, 16), (3, 64, 64)])
# summaries(model, grad=True)

PLoss = nn.L1Loss(size_average=True).cuda()
# Sparse_loss = SparseKLloss().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)   # optimizer 1
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)  # optimizer 2
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=200,
                                              gamma=0.1)  # lr = lr* 1/gamma for each step_size = 180

# if os.path.exists('train_logs'):  # for tensorboard: copy dir of train_logs
#     shutil.rmtree('train_logs')  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs  --host=127.0.0.1
# writer = SummaryWriter('./train_logs(model-Trans)/')

model_folder = "./Ours_Trained_model/"

def save_checkpoint(model, epoch):  # save model function

    model_out_path = model_folder + "{}.pth".format(epoch)

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "lr":lr
    }
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    torch.save(checkpoint, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################

def train(training_data_loader, validate_data_loader,start_epoch=0,RESUME=False):
    import matplotlib.pyplot as plt
    plt.ion()
    fig, axes = plt.subplots(ncols=2, nrows=2)
    print('Start training...')

    if RESUME:
        path_checkpoint = model_folder+"{}.pth".format(500)
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Network is Successfully Loaded from %s' % (path_checkpoint))
    time_s = time.time()
    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            GT, LRHSI  = batch[0].cuda(), batch[1].cuda()
            LRHSI = LRHSI.cuda()
            GT = GT.cuda()

            optimizer.zero_grad()  # fixed
            

            # output_HRHSI, total_spec_loss, total_boundary_loss = model(LRHSI, torch.zeros(batch_size, 1, 1, 1).cuda())
            output_HRHSI3,output_HRHSI4,output_HRHSI5,output_HRHSI6,output_HRHSI7 = model(LRHSI)
            # output_HRHSI = F.interpolate(output_HRHSI, scale_factor=(4, 4),mode ='bicubic')
            time_e = time.time()
            # Pixelwise_Loss =PLoss(output_HRHSI, GT)+0.1*total_spec_loss+0.1*total_boundary_loss
            Pixelwise_Loss =0.5*(loss(output_HRHSI3, GT) + loss(output_HRHSI4, GT) + loss(output_HRHSI5, GT) + loss(output_HRHSI6, GT)) + loss(output_HRHSI7, GT)  
            Myloss = Pixelwise_Loss
            epoch_train_loss.append(Myloss.item())  # save all losses into a vector for one epoch

            Myloss.backward()  # fixed
            optimizer.step()  # fixed

            if iteration % 10 == 0:
                # log_value('Loss', loss.data[0], iteration + (epoch - 1) * len(training_data_loader))
                print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader),
                                                                   Myloss.item()))
        print("learning rate:º%f" % (optimizer.param_groups[0]['lr']))
        lr_scheduler.step()  # update lr

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss

        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch
        print(time_e - time_s)
        
        # ============Epoch Validate=============== #
        if epoch % 1== 0:
            model.eval()
            with torch.no_grad():
                for iteration, batch in enumerate(validate_data_loader, 1):
                    GT, LRHSI  = batch[0].cuda(), batch[1].cuda() 
  
                ref = GT.detach().cpu().numpy()
                out = output_HRHSI7.detach().cpu().numpy()

                psnr = calc_psnr(ref, out)
                rmse = calc_rmse(ref, out)
                ergas = calc_ergas(ref, out)
                sam = calc_sam(ref, out)
                Ssim = ssim(ref, out)

                print('RMSE:   {:.4f};'.format(rmse))
                print('PSNR:   {:.4f};'.format(psnr))
                print('ERGAS:   {:.4f};'.format(ergas))
                print('SAM:   {:.4f}.'.format(sam))
                print('SSIM:   {:.4f}.'.format(Ssim))

                with open("metric_txt.txt","a+") as f:
                    f.write(str(epoch)+'\n')
                    f.write(str(psnr)+'\n')
                    f.write(str(rmse)+'\n')
                    f.write(str(ergas)+'\n')
                    f.write(str(sam)+'\n')
                    f.write(str(Ssim)+'\n')
                    f.close


            if epoch % ckpt_step == 0:  # if each ckpt epochs, then start to save model
                save_checkpoint(model, epoch)

def test():
    test_set = DatasetFromHdf5("G:/Hyperspectral image super-resolution/Experiments/Train_and_test_Datasets/Chikusei_Train_test/test_chikusei_64_256_scale_4.h5")
   # test_set = DatasetFromHdf5("G:/Hyperspectral image super-resolution/Experiments/Train_and_test_Datasets/Pavia_test/test_pavia_64_256_scale_4.h5")
    # test_set = DatasetFromHdf5("G:/Hyperspectral image super-resolution/Experiments/Train_and_test_Datasets/Houston_2013_test/test_HoustonU_2018_128_512_scale_4.h5")
    print(torch.cuda.get_device_name(0))
    num_testing = 8
    # num_testing = 8
    # num_testing = 8
    testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1)
    sz = 256
    output_HRHSI = np.zeros((num_testing, sz, sz, 128))
    UP_LRHSI = np.zeros((num_testing, sz, sz, 128))
    
    # output_HRHSI = np.zeros((num_testing, sz, sz, 102))
    # UP_LRHSI = np.zeros((num_testing, sz, sz, 102))
    # output_HRHSI = np.zeros((num_testing, sz, sz, 48))
    # UP_LRHSI = np.zeros((num_testing, sz, sz, 48))
    # model = EUNet(2, 4, 128, n_feats=128, n_modules=3, n_blocks=2, dilations=[1,2], expand_ratio=2).cuda()
    model = HSI_ReNet_g(128, 4, 20).cuda()
    #model = torch.nn.DataParallel(model)
    #path_checkpoint = model_folder + "{}.pth".format(1000)
    path_checkpoint = model_folder + "100.pth"
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['net'])
    model = model.cuda()
    #model = nn.DataParallel(model)
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size(), parameters)
    psnr_list = []
    sam_list = []
    ergas_list = []
    rmse_list = []
    ssim_list = []


    for iteration, batch in enumerate(testing_data_loader, 1):
        GT, LRHSI = Variable(batch[0]), Variable(batch[1])
        LRHSI = LRHSI.cuda()
        GT = GT.cuda()

        with torch.no_grad():
            # output_HRHSIone = model(LRHSI, torch.zeros(1, 1, 1, 1).cuda())
            _,_,_,_, output_HRHSIone = model(LRHSI)
            # output_HRHSIone = F.interpolate(LRHSI, scale_factor=(2, 2),mode ='bicubic')

        #output_HRHSIbic = F.interpolate(LRHSI, scale_factor=(16, 16),mode ='bicubic')
        #out1 = output_HRHSIbic.detach().cpu().numpy()

        ref = GT.detach().cpu().numpy()
        out1 = output_HRHSIone.detach().cpu().numpy()

        psnr = calc_psnr(ref, out1)
        sam = calc_sam(ref, out1)
        ergas = calc_ergas(ref, out1)
        rmse = calc_rmse(ref, out1)
        Ssim = ssim(ref, out1)

        psnr_list.append(psnr)
        sam_list.append(sam)
        ergas_list.append(ergas)
        rmse_list.append(rmse)
        ssim_list.append(Ssim)

        output_HRHSI[iteration-1,:,:,:] = output_HRHSIone.permute([0, 3, 2, 1]).cpu().detach().numpy()
        #UP_LRHSI[iteration-1,:,:,:] = UP_LRHSIone.permute([0, 2, 3, 1]).cpu().detach().numpy()
        #Highpass[iteration - 1, :, :, :] = Highpassone.permute([0, 2, 3, 1]).cpu().detach().numpy()

    print('{:.4f}'.format(np.array(psnr_list).mean()))
    print('{:.4f}'.format(np.array(sam_list).mean()))
    print('{:.4f}'.format(np.array(ergas_list).mean()))
    print('{:.4f}'.format(np.array(rmse_list).mean()))
    print('{:.4f}'.format(np.array(ssim_list).mean()))
    # sio.savemat('Ours_Houston_scale_4.mat',{'Ours_Houston_scale_4': output_HRHSI})
###################################################################
# ------------------- Main Function  -------------------
###################################################################
if __name__ == "__main__":
    train_or_not =0
    test_or_not =1

    if train_or_not:
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.device_count())
        train_set = DatasetFromHdf5("G:/Hyperspectral image super-resolution/Experiments/Train_and_test_Datasets/Chikusei_Train_test/train_chikusei_8_32_scale_4_small.h5")  # creat data for training
        training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                          pin_memory=True, drop_last=True)  # put training data to DataLoader for batches
        validate_set = DatasetFromHdf5("G:/Hyperspectral image super-resolution/Experiments/Train_and_test_Datasets/Chikusei_Train_test/test_chikusei_64_256_scale_4.h5")  # creat data for validation
        validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                          pin_memory=True, drop_last=True)  # put training data to DataLoader for batches
        train(training_data_loader, validate_data_loader)#, start_epoch=200)  # call train function (call: Line 53)

    if test_or_not:
        print("----------------------------testing-------------------------------")
        test()