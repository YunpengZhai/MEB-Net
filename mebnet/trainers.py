from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
import random
from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from .utils.meters import AverageMeter


class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            # target samples: only forward
            t_features, _ = self.model(t_inputs)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


class SingelmeanTrainer(object):
    def __init__(self, model_1, 
                       model_1_ema, num_cluster=500, alpha=0.999):
        super(SingelmeanTrainer, self).__init__()
        self.model_1 = model_1
        self.num_cluster = num_cluster

        self.model_1_ema = model_1_ema
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch, data_loader_target,
            optimizer, ce_soft_weight=0.5, tri_soft_weight=0.5, print_freq=1, train_iters=200):
        self.model_1.train()
        self.model_1_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_1, targets = self._parse_data(target_inputs)

            # forward
            f_out_t1, p_out_t1 = self.model_1(inputs_1)

            f_out_t1_ema, p_out_t1_ema = self.model_1_ema(inputs_1)

            loss_ce_1 = self.criterion_ce(p_out_t1, targets)

            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, targets)

            loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t1_ema)

            loss_tri_soft = self.criterion_tri_soft(f_out_t1, p_out_t1_ema, targets)

            loss = loss_ce_1*(1-ce_soft_weight) + \
                     loss_tri_1*(1-tri_soft_weight) + \
                     loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(p_out_t1.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_tri[0].update(loss_tri_1.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            precisions[0].update(prec_1[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f}\t'
                      'Loss_tri {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Prec {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg,
                              losses_tri[0].avg, 
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              precisions[0].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        # import pdb;pdb.set_trace()
        imgs_1, imgs_2, pids = inputs
        inputs_1 = imgs_1.cuda()
        targets = pids.cuda()
        return inputs_1, targets


# EBS + preference scatter
class MEBTrainer(object):
    def __init__(self, model_list, model_ema_list, num_cluster=500, alpha=0.999, scatter=[1,1,1]):
        super(MEBTrainer, self).__init__()
        self.models = model_list
        self.num_cluster = num_cluster
        self.model_num = len(self.models)
        self.model_emas = model_ema_list
        self.alpha = alpha
        self.scatter = F.normalize(torch.FloatTensor(scatter).cuda() ,p=1, dim=0)

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch, data_loader_target,
            optimizer, ce_soft_weight=0.5, tri_soft_weight=0.5, print_freq=1, train_iters=200):
        for model in self.models:
            model.train()
        for model_ema in self.model_emas:
            model_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precisions = [AverageMeter() for i in range(self.model_num)]

        end = time.time()
        for iter_idx in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets = self._parse_data(target_inputs)

            # forward
            f_out_t = []
            p_out_t = []
            for i in range(self.model_num):
                f_out_t_i, p_out_t_i = self.models[i](inputs[i])
                f_out_t.append(f_out_t_i)
                p_out_t.append(p_out_t_i)

            f_out_t_ema = []
            p_out_t_ema = []
            for i in range(self.model_num):
                f_out_t_ema_i, p_out_t_ema_i = self.model_emas[i](inputs[i])
                f_out_t_ema.append(f_out_t_ema_i)
                p_out_t_ema.append(p_out_t_ema_i)

            
            authority_ce = []
            authority_tri = []

            loss_ce = loss_tri = 0
            for i in range(self.model_num):
                loss_ce += self.criterion_ce(p_out_t[i], targets)
                loss_tri += self.criterion_tri(f_out_t[i], f_out_t[i], targets)

                authority_ce.append(self.criterion_ce(p_out_t[i], targets))
                authority_tri.append(self.criterion_tri(f_out_t[i], f_out_t[i], targets))
            
            beta = 2
            authority = 3*self.scatter

            
            loss_ce_soft = loss_tri_soft = 0
            for i in range(self.model_num): #speaker
                for j in range(self.model_num): #listener
                    if i != j:
                        loss_ce_soft += 0.5*authority[i]*self.criterion_ce_soft(p_out_t[j], p_out_t_ema[i])
                        loss_tri_soft += 0.5*authority[i]*self.criterion_tri_soft(f_out_t[j], f_out_t_ema[i], targets)

            loss = loss_ce*(1-ce_soft_weight) + \
                     loss_tri*(1-tri_soft_weight) + \
                     loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for i in range(self.model_num):
                self._update_ema_variables(self.models[i], self.model_emas[i], self.alpha, epoch*len(data_loader_target)+iter_idx)

            prec_1, = accuracy(p_out_t[0].data, targets.data)
            prec_2, = accuracy(p_out_t[1].data, targets.data)
            prec_3, = accuracy(p_out_t[2].data, targets.data)

            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            precisions[0].update(prec_1[0])
            precisions[1].update(prec_2[0])
            precisions[2].update(prec_3[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (iter_idx + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f}\t'
                      'Loss_tri {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Prec {:.2%} / {:.2%} / {:.2%}\t'
                      .format(epoch, iter_idx + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.avg, losses_tri.avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              precisions[0].avg, precisions[1].avg, precisions[2].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, imgs_3, pids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        inputs_3 = imgs_3.cuda()
        targets = pids.cuda()
        inputs_list = [inputs_1,inputs_2,inputs_3]
        return inputs_list, targets
