import argparse
import os
from os.path import join

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data import PancreasCancerDatasetAugmentationFoldReverse
from model.GCN import GCN_Module
from model.Meta_Dense_Unet import get_updated_network_new, Encoder_Dense, Decoder_Dense
from model.Transfer_Module import Private_Encoder, Private_Decoder
from model.model_gan import NLayerDiscriminator, GANLoss
from util.loss import VGGLoss, VGGLoss_for_trans
from util.utils import (
    adjust_learning_rate,
    UpsampleDeterministic,
    convertToMultiChannel,
    DICELoss_LV,
    seed_torch,
    save_arg,
    get_time,
)

seed_torch(2019)


def get_args():
    parser = argparse.ArgumentParser(description="UNet for Pancreas cancer Dataset")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for training (default: 1)",
    )
    parser.add_argument(
        "--meta", type=int, default=5, metavar="N", help="meta (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2019,
        metavar="N",
        help="random seed (default: 2019)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for testing (default: 1)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=True,
        help="Argument to train model (default: False)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 12)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="denseunetcbam",
        metavar="str",
        help="model to load (default: denseunetcbam)",
    )
    parser.add_argument(
        "--r1",
        type=float,
        default=1,
        metavar="R1",
        help="lamda 1 for discriminator (default: 1)",
    )
    parser.add_argument(
        "--r2",
        type=float,
        default=1,
        metavar="R2",
        help="lamda 2 for discriminator (default: 1)",
    )
    parser.add_argument(
        "--adam-lr",
        type=float,
        default=0.0002,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--pow", type=float, default=0.9, metavar="POW", help="pow rate (default: 0.9)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0,
        metavar="dr",
        help="dropout rate (default: 0)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="enables CUDA training (default: true)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="batches to wait before logging training status",
    )
    parser.add_argument("--size", type=int, default=256, metavar="N", help="imsize")
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        metavar="str",
        help="weight file to load (default: None)",
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        default="./Data/",
        metavar="str",
        help="folder that contains data (default: test dataset)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="time",
        metavar="str",
        help="time for identify file (default: time)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="OutMasks",
        metavar="str",
        help="Identifier to save npy arrays with",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        metavar="N",
        help="input visible devices for training (default: 0)",
    )

    parser.add_argument(
        "--modality",
        type=str,
        default="flair",
        metavar="str",
        help="Modality to use for training (default: flair)",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        default=False,
        help="enables GCN Weight (default: False)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        metavar="str",
        help="Optimizer (default: SGD)",
    )
    parser.add_argument(
        "--dis",
        type=str,
        default="NL",
        metavar="str",
        help="Discriminator (default: NL)",
    )
    parser.add_argument("--fold", type=int, default=0, metavar="N", help="fold(0-3)")
    parser.add_argument(
        "--random-index", type=int, default=0, metavar="N", help="--random-index(0-19)"
    )
    return parser.parse_args()


def train():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("POW:", args.pow)

    epochs = args.epochs
    learning_rate_seg = 0.01
    learning_rate_d = 1e-4
    learning_rate_rec = 1e-3
    learning_rate_dis = 1e-4
    power = args.pow
    power_g = args.pow
    weight_decay = 0.0005
    lambda_adv_target1 = 0.0002
    lambda_adv_target2 = 0.001
    loss_lambda = [1, 0.5, 0.01, 0.05, 0.1]
    outer_stepsize_maml = 0.01
    cur_time = args.time
    if cur_time == "time":
        cur_time = get_time()

    dset_train = PancreasCancerDatasetAugmentationFoldReverse(fold=args.fold)
    train_loader = DataLoader(
        dset_train, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    model_dict = {}

    # Setup Model
    print("building models ...")

    enc_shared = Encoder_Dense().cuda()
    dec_shared = Decoder_Dense().cuda()
    updated_encoder = Encoder_Dense().cuda()
    updated_decoder = Decoder_Dense().cuda()

    dclf1 = NLayerDiscriminator(input_nc=512).cuda()
    dclf2 = NLayerDiscriminator(input_nc=512).cuda()
    enc_s = Private_Encoder(64).cuda()
    enc_t = Private_Encoder(64).cuda()
    dec_s = Private_Decoder(512).cuda()
    dec_t = dec_s
    dis_s2t = NLayerDiscriminator().cuda()
    dis_t2s = NLayerDiscriminator().cuda()
    GCN = GCN_Module(512, args.weighted).cuda()

    model_dict["enc_shared"] = enc_shared
    model_dict["dec_shared"] = dec_shared
    model_dict["dclf1"] = dclf1
    model_dict["dclf2"] = dclf2
    model_dict["enc_s"] = enc_s
    model_dict["enc_t"] = enc_t
    model_dict["dec_s"] = dec_s
    model_dict["dec_t"] = dec_t
    model_dict["dis_s2t"] = dis_s2t
    model_dict["dis_t2s"] = dis_t2s

    dclf1.train()
    dclf2.train()
    enc_shared.train()
    dec_shared.train()
    enc_s.train()
    enc_t.train()
    dec_s.train()
    dec_t.train()
    dis_s2t.train()
    dis_t2s.train()
    GCN.train()

    enc_shared_opt = optim.SGD(
        enc_shared.parameters(),
        lr=learning_rate_seg,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    dec_shared_opt = optim.SGD(
        dec_shared.parameters(),
        lr=learning_rate_seg,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    optimizer_gcn = optim.SGD(GCN.parameters(), lr=args.lr, momentum=0.99)

    dclf1_opt = optim.Adam(dclf1.parameters(), lr=learning_rate_d, betas=(0.9, 0.99))
    dclf2_opt = optim.Adam(dclf2.parameters(), lr=learning_rate_d, betas=(0.9, 0.99))
    enc_s_opt = optim.Adam(enc_s.parameters(), lr=learning_rate_rec, betas=(0.5, 0.999))
    enc_t_opt = optim.Adam(enc_t.parameters(), lr=learning_rate_rec, betas=(0.5, 0.999))
    dec_s_opt = optim.Adam(dec_s.parameters(), lr=learning_rate_rec, betas=(0.5, 0.999))
    dec_t_opt = optim.Adam(dec_t.parameters(), lr=learning_rate_rec, betas=(0.5, 0.999))
    dis_s2t_opt = optim.Adam(
        dis_s2t.parameters(), lr=learning_rate_dis, betas=(0.5, 0.999)
    )
    dis_t2s_opt = optim.Adam(
        dis_t2s.parameters(), lr=learning_rate_dis, betas=(0.5, 0.999)
    )

    seg_opt_list = []
    dclf_opt_list = []
    rec_opt_list = []
    dis_opt_list = []

    # Optimizer list for quickly adjusting learning rate
    seg_opt_list.append(enc_shared_opt)
    seg_opt_list.append(dec_shared_opt)
    seg_opt_list.append(optimizer_gcn)
    dclf_opt_list.append(dclf1_opt)
    dclf_opt_list.append(dclf2_opt)
    rec_opt_list.append(enc_s_opt)
    rec_opt_list.append(enc_t_opt)
    rec_opt_list.append(dec_s_opt)
    rec_opt_list.append(dec_t_opt)
    dis_opt_list.append(dis_s2t_opt)
    dis_opt_list.append(dis_t2s_opt)

    sg_loss = DICELoss_LV()
    criterion_gan = GANLoss(use_lsgan=True).cuda()
    VGG_loss = VGGLoss()
    VGG_loss_for_trans = VGGLoss_for_trans()
    criterion_gcn = nn.MSELoss().cuda()
    pool = nn.AvgPool2d(16).cuda()

    print("#################################")
    print("fold:", args.fold)
    print("epoch:", args.epochs)
    print("batch_size:", args.batch_size)
    print("#################################")

    save_arg(args, "t2_to_t1_F%s_%s.txt" % (args.fold, cur_time))

    for i_iter in range(epochs):
        loss_list = []
        enc_shared.train()
        dec_shared.train()
        adjust_learning_rate(
            seg_opt_list,
            base_lr=learning_rate_seg,
            i_iter=i_iter,
            max_iter=args.epochs,
            power=power_g,
        )
        adjust_learning_rate(
            dclf_opt_list,
            base_lr=learning_rate_d,
            i_iter=i_iter,
            max_iter=args.epochs,
            power=power,
        )
        adjust_learning_rate(
            rec_opt_list,
            base_lr=learning_rate_rec,
            i_iter=i_iter,
            max_iter=args.epochs,
            power=power,
        )
        adjust_learning_rate(
            dis_opt_list,
            base_lr=learning_rate_dis,
            i_iter=i_iter,
            max_iter=args.epochs,
            power=power,
        )

        with tqdm(train_loader) as t:
            for (
                batch_idx,
                (source_data, source_label, target_data, target_label),
            ) in enumerate(t):
                t.set_description("epoch %s" % i_iter)
                sdatav = Variable(source_data.float()).cuda()
                slabelv = Variable(source_label.float()).cuda()
                tdatav = Variable(target_data.float()).cuda()

                code_s_common, s_pred3, s_pred2, s_pred1, low_s = enc_shared(sdatav)
                code_t_common, t_pred3, t_pred2, t_pred1, low_t = enc_shared(tdatav)
                code_s_private = enc_s(low_s)
                code_t_private = enc_t(low_t)

                rec_s = dec_s(code_s_common, code_s_private, 0)
                rec_t = dec_t(code_t_common, code_t_private, 1)
                rec_t2s = dec_s(code_t_common, code_s_private, 0)
                rec_s2t = dec_t(code_s_common, code_t_private, 1)

                for p in dclf1.parameters():
                    p.requires_grad = True
                for p in dclf2.parameters():
                    p.requires_grad = True
                for p in dis_s2t.parameters():
                    p.requires_grad = True
                for p in dis_t2s.parameters():
                    p.requires_grad = True

                # ===== dclf1 =====
                prob_dclf1_real1 = dclf1(
                    UpsampleDeterministic(upscale=16)(code_s_common.detach())
                )
                prob_dclf1_fake1 = dclf1(
                    UpsampleDeterministic(upscale=16)(code_t_common.detach())
                )
                loss_d_dclf1 = (
                    0.5 * criterion_gan(prob_dclf1_real1, True).cuda()
                    + 0.5 * criterion_gan(prob_dclf1_fake1, False).cuda()
                )

                dclf1_opt.zero_grad()
                loss_d_dclf1.backward()
                dclf1_opt.step()

                # ===== dclf2 =====
                prob_dclf2_real1 = dclf2(
                    UpsampleDeterministic(upscale=8)(s_pred3.detach())
                )
                prob_dclf2_fake1 = dclf2(
                    UpsampleDeterministic(upscale=8)(t_pred3.detach())
                )
                loss_d_dclf2 = (
                    0.5 * criterion_gan(prob_dclf2_real1, True).cuda()
                    + 0.5 * criterion_gan(prob_dclf2_fake1, False).cuda()
                )

                dclf2_opt.zero_grad()
                loss_d_dclf2.backward()
                dclf2_opt.step()

                # clip parameters in D
                for p in dclf1.parameters():
                    p.data.clamp_(-0.05, 0.05)

                for p in dclf2.parameters():
                    p.data.clamp_(-0.05, 0.05)

                # train image discriminator
                # ===== dis_s2t =====
                if (i_iter + 1) % 5 == 0:
                    prob_dis_s2t_real1 = dis_s2t(tdatav)
                    prob_dis_s2t_fake1 = dis_s2t(rec_s2t.detach())
                    loss_d_s2t = (
                        0.5 * criterion_gan(prob_dis_s2t_real1, True).cuda()
                        + 0.5 * criterion_gan(prob_dis_s2t_fake1, False).cuda()
                    )
                    dis_s2t_opt.zero_grad()
                    loss_d_s2t.backward()
                    dis_s2t_opt.step()

                # ===== dis_t2s =====
                if (i_iter + 1) % 5 == 0:
                    prob_dis_t2s_real1 = dis_t2s(sdatav)
                    prob_dis_t2s_fake1 = dis_t2s(rec_t2s.detach())
                    loss_d_t2s = (
                        0.5 * criterion_gan(prob_dis_t2s_real1, True).cuda()
                        + 0.5 * criterion_gan(prob_dis_t2s_fake1, False).cuda()
                    )

                    dis_t2s_opt.zero_grad()
                    loss_d_t2s.backward()
                    dis_t2s_opt.step()

                for p in dis_s2t.parameters():
                    p.data.clamp_(-0.05, 0.05)

                for p in dis_t2s.parameters():
                    p.data.clamp_(-0.05, 0.05)

                for p in dclf1.parameters():
                    p.requires_grad = False
                for p in dclf2.parameters():
                    p.requires_grad = False
                for p in dis_s2t.parameters():
                    p.requires_grad = False
                for p in dis_t2s.parameters():
                    p.requires_grad = False

                # not necessary
                # ==== VGGLoss self-reconstruction loss ====
                loss_rec_s = VGG_loss(
                    convertToMultiChannel(rec_s), convertToMultiChannel(sdatav)
                )
                loss_rec_t = VGG_loss(
                    convertToMultiChannel(rec_t), convertToMultiChannel(tdatav)
                )
                loss_rec_self = loss_rec_s + loss_rec_t

                loss_rec_s2t = VGG_loss_for_trans(
                    convertToMultiChannel(rec_s2t),
                    convertToMultiChannel(sdatav),
                    convertToMultiChannel(tdatav),
                    weights=[0, 0, 0, 1.0 / 4, 1.0],
                )
                loss_rec_t2s = VGG_loss_for_trans(
                    convertToMultiChannel(rec_t2s),
                    convertToMultiChannel(tdatav),
                    convertToMultiChannel(sdatav),
                    weights=[0, 0, 0, 1.0 / 4, 1.0],
                )
                loss_rec_tran = loss_rec_s2t + loss_rec_t2s

                loss_rec = loss_rec_tran + loss_rec_self * 6

                # ==== domain agnostic loss ====
                prob_dclf1_fake2 = dclf1(
                    UpsampleDeterministic(upscale=16)(code_t_common)
                )
                loss_feat1_similarity = criterion_gan(prob_dclf1_fake2, True)

                prob_dclf2_fake2 = dclf2(UpsampleDeterministic(upscale=8)(t_pred3))
                loss_feat2_similarity = criterion_gan(prob_dclf2_fake2, True)

                loss_feat_similarity = (
                    lambda_adv_target1 * loss_feat1_similarity
                    + lambda_adv_target2 * loss_feat2_similarity
                )

                # ==== image translation loss ====
                prob_dis_s2t_fake2 = dis_s2t(rec_s2t)
                loss_gen_s2t = criterion_gan(prob_dis_s2t_fake2, True)
                prob_dis_t2s_fake2 = dis_t2s(rec_t2s)
                loss_gen_t2s = criterion_gan(prob_dis_t2s_fake2, True)
                loss_image_translation = loss_gen_s2t + loss_gen_t2s

                # ==== segmentation loss ====
                if args.meta == 0:
                    iter_if = i_iter == 0
                else:
                    iter_if = i_iter % args.meta == 0

                if iter_if:
                    # linear schedule
                    outer_step_size = outer_stepsize_maml * (1 - i_iter / epochs)

                    ####################### inter iter #######################
                    enc_shared.zero_grad()
                    dec_shared.zero_grad()

                    s_pred1 = dec_shared(
                        code_s_common, s_pred3, s_pred2, s_pred1, low_s
                    )
                    loss_sim_sg = sg_loss(s_pred1, slabelv)
                    loss_list.append(loss_sim_sg.item())

                    gcn_output = GCN(code_s_common)
                    gcn_label = pool(slabelv)
                    loss_gcn = criterion_gcn(gcn_output, gcn_label)

                    loss_sim_sg += 0.1 * loss_gcn
                    loss_sim_sg.backward(retain_graph=True)

                    ####################### outer iter #######################
                    updated_encoder = (
                        get_updated_network_new(
                            enc_shared, updated_encoder, outer_step_size
                        )
                        .train()
                        .cuda()
                    )
                    updated_decoder = (
                        get_updated_network_new(
                            dec_shared, updated_decoder, outer_step_size
                        )
                        .train()
                        .cuda()
                    )

                    s2t_code_s_common, s2t_s_pred3, s2t_s_pred2, s2t_s_pred1, s2t_low_s = updated_encoder(
                        rec_s2t
                    )
                    s2t_pred1 = updated_decoder(
                        s2t_code_s_common,
                        s2t_s_pred3,
                        s2t_s_pred2,
                        s2t_s_pred1,
                        s2t_low_s,
                    )
                    loss_sim_sg_s2t = sg_loss(s2t_pred1, slabelv)

                    gcn_output_s2t = GCN(s2t_code_s_common)
                    loss_gcn = criterion_gcn(gcn_output_s2t, gcn_label)

                    total_loss = (
                        loss_lambda[0] * loss_sim_sg_s2t
                        + loss_lambda[1] * loss_feat_similarity
                        + loss_lambda[2] * loss_image_translation
                        + loss_lambda[3] * loss_rec
                        + loss_lambda[4] * loss_gcn
                    ) + loss_sim_sg

                    enc_s_opt.zero_grad()
                    enc_t_opt.zero_grad()
                    dec_s_opt.zero_grad()
                    optimizer_gcn.zero_grad()

                    total_loss.backward()

                    enc_s_opt.step()
                    enc_t_opt.step()
                    dec_s_opt.step()
                    optimizer_gcn.step()
                    enc_shared_opt.step()
                    dec_shared_opt.step()

                else:
                    s_pred1 = dec_shared(
                        code_s_common, s_pred3, s_pred2, s_pred1, low_s
                    )
                    loss_sim_sg = sg_loss(s_pred1, slabelv)
                    loss_list.append(loss_sim_sg.item())

                    gcn_output = GCN(code_s_common)
                    gcn_label = pool(slabelv)
                    loss_gcn = criterion_gcn(gcn_output, gcn_label)

                    s2t_code_s_common, s2t_s_pred3, s2t_s_pred2, s2t_s_pred1, s2t_low_s = enc_shared(
                        rec_s2t
                    )
                    s2t_pred1 = dec_shared(
                        s2t_code_s_common,
                        s2t_s_pred3,
                        s2t_s_pred2,
                        s2t_s_pred1,
                        s2t_low_s,
                    )
                    loss_sim_sg += sg_loss(s2t_pred1, slabelv)

                    gcn_output_s2t = GCN(s2t_code_s_common)
                    loss_gcn = loss_gcn + criterion_gcn(gcn_output_s2t, gcn_label)

                    total_loss = (
                        loss_lambda[0] * loss_sim_sg
                        + loss_lambda[1] * loss_feat_similarity
                        + loss_lambda[2] * loss_image_translation
                        + loss_lambda[3] * loss_rec
                        + loss_lambda[4] * loss_gcn
                    )

                    enc_shared_opt.zero_grad()
                    dec_shared_opt.zero_grad()
                    enc_s_opt.zero_grad()
                    enc_t_opt.zero_grad()
                    dec_s_opt.zero_grad()
                    optimizer_gcn.zero_grad()

                    total_loss.backward()

                    enc_shared_opt.step()
                    dec_shared_opt.step()
                    enc_s_opt.step()
                    enc_t_opt.step()
                    dec_s_opt.step()
                    optimizer_gcn.step()

                t.set_postfix(Ave_Seg_Loss=np.mean(loss_list), cur_loss=loss_list[-1])

        if (i_iter + 1) % 1 == 0:

            with open("logs/t2_to_t1_F%s_%s.txt" % (args.fold, cur_time), "a") as f:
                f.write("ave_loss=%s \n" % (np.mean(loss_list)))

    torch.save(enc_shared.state_dict(), join("checkpoints", "t2_to_t1_enc_shared.pth"))
    torch.save(dec_shared.state_dict(), join("checkpoints", "t2_to_t1_dec_shared.pth"))


if __name__ == "__main__":
    train()
