from copy import deepcopy

import torch
from timm.models.layers import DropPath
from torch import nn
from torch.nn.modules.dropout import _DropoutNd


class EMATeacher(nn.Module):

    def __init__(self, model,classifier,classifier_p, alpha, pseudo_label_weight):
        super(EMATeacher, self).__init__()
        self.ema_model = deepcopy(model)
        self.f = deepcopy(classifier)
        self.f_p = deepcopy(classifier_p)
        self.alpha = alpha
        self.pseudo_label_weight = pseudo_label_weight
        if self.pseudo_label_weight == 'None':
            self.pseudo_label_weight = None

    def _init_ema_weights(self, model,classifier):
        # ema model
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.ema_model.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()
        # f
        for param in self.f.parameters():
            param.detach_()
        mp = list(classifier.parameters())
        mcp = list(self.f.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

        # f_p
        for param in self.f_p.parameters():
            param.detach_()
        mp = list(classifier.parameters())
        mcp = list(self.f_p.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model,classifier,classifier_p, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
        for ema_param, param in zip(self.f.parameters(),
                                    classifier.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

        for ema_param, param in zip(self.f_p.parameters(),
                                    classifier_p.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
    def update_weights(self, model,classifier,classifier_p, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model,classifier,classifier_p)
        if iter > 0:
            self._update_ema(model,classifier,classifier_p, iter)

    @torch.no_grad()
    def forward(self, target_img):
        # Generate pseudo-label
        for m in self.ema_model.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        target_img,target_img_p,target_img_k = self.ema_model(target_img)#_(32,64)

        logits , _ = self.f(target_img)
        logits_p, _ = self.f_p(target_img_p)

        ema_softmax = torch.softmax(logits.detach(), dim=1)#.detach() 将 logits 从计算图中分离出来，防止梯度回传到生成这些 logits 的网络。
        # 这在生成伪标签时非常重要，因为伪标签不应影响教师模型的更新。
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)#获取每个样本的最大概率值及其对应的索引。
        ema_softmax_p = torch.softmax(logits_p.detach(), dim=1)  # .detach() 将 logits 从计算图中分离出来，防止梯度回传到生成这些 logits 的网络。
        # 这在生成伪标签时非常重要，因为伪标签不应影响教师模型的更新。
        pseudo_prob_p, pseudo_label_p = torch.max(ema_softmax_p, dim=1)

        if self.pseudo_label_weight is None:
            pseudo_weight = torch.tensor(1., device=logits.device)
            pseudo_weight_p = torch.tensor(1., device=logits_p.device)
        elif self.pseudo_label_weight == 'prob':
            pseudo_weight = pseudo_prob
            pseudo_weight_p = pseudo_prob_p
        else:
            raise NotImplementedError(self.pseudo_label_weight)

        return pseudo_label, pseudo_weight, pseudo_label_p, pseudo_weight_p

class EMATeacher_t(nn.Module):

    def __init__(self, model,classifier,text_encode,alpha, pseudo_label_weight):
        super(EMATeacher, self).__init__()
        self.ema_model = deepcopy(model)
        self.f=deepcopy(classifier)
        self.t = deepcopy(text_encode)
        self.alpha = alpha
        self.pseudo_label_weight = pseudo_label_weight
        if self.pseudo_label_weight == 'None':
            self.pseudo_label_weight = None

    def _init_ema_weights(self, model,classifier, text_encode):
        # ema model
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.ema_model.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()
        # f
        for param in self.f.parameters():
            param.detach_()
        mp = list(classifier.parameters())
        mcp = list(self.f.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()
        #text_encode
        for param in self.t.parameters():
            param.detach_()
        mp = list(text_encode.parameters())
        mcp = list(self.t.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model,classifier,text_encode, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
        for ema_param, param in zip(self.f.parameters(),
                                    classifier.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
        for ema_param, param in zip(self.t.parameters(),
                                    text_encode.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def update_weights(self, model,classifier,text_encode, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model,classifier,text_encode)
        if iter > 0:
            self._update_ema(model,classifier,text_encode, iter)

    @torch.no_grad()
    def forward(self, target_img):
        # Generate pseudo-label
        for m in self.ema_model.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        _ = self.ema_model(target_img)#_(32,64)

        logits , _ = self.f(_)
    #文本编码器没有加上去
        ema_softmax = torch.softmax(logits.detach(), dim=1)#.detach() 将 logits 从计算图中分离出来，防止梯度回传到生成这些 logits 的网络。
        # 这在生成伪标签时非常重要，因为伪标签不应影响教师模型的更新。
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)#获取每个样本的最大概率值及其对应的索引。

        if self.pseudo_label_weight is None:
            pseudo_weight = torch.tensor(1., device=logits.device)
        elif self.pseudo_label_weight == 'prob':
            pseudo_weight = pseudo_prob
        else:
            raise NotImplementedError(self.pseudo_label_weight)

        return pseudo_label, pseudo_weight
class EMATeacherfft(nn.Module):

    def __init__(self, model1,model2,classifier, alpha, pseudo_label_weight):
        super(EMATeacherfft, self).__init__()
        self.ema_model1 = deepcopy(model1)
        self.ema_model2 = deepcopy(model2)
        self.f=deepcopy(classifier)
        self.alpha = alpha
        self.pseudo_label_weight = pseudo_label_weight
        if self.pseudo_label_weight == 'None':
            self.pseudo_label_weight = None

    def _init_ema_weights(self, model,classifier):
        # ema model
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.ema_model.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()
        # f
        for param in self.f.parameters():
            param.detach_()
        mp = list(classifier.parameters())
        mcp = list(self.f.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model,classifier, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
        for ema_param, param in zip(self.f.parameters(),
                                    classifier.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def update_weights(self, model,classifier, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model,classifier)
        if iter > 0:
            self._update_ema(model,classifier, iter)

    @torch.no_grad()
    def forward(self, target_img):
        # Generate pseudo-label
        for m in self.ema_model.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        _ = self.ema_model(target_img)
        logits , _ = self.f(_)

        ema_softmax = torch.softmax(logits.detach(), dim=1)#.detach() 将 logits 从计算图中分离出来，防止梯度回传到生成这些 logits 的网络。
        # 这在生成伪标签时非常重要，因为伪标签不应影响教师模型的更新。
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)#获取每个样本的最大概率值及其对应的索引。

        if self.pseudo_label_weight is None:
            pseudo_weight = torch.tensor(1., device=logits.device)
        elif self.pseudo_label_weight == 'prob':
            pseudo_weight = pseudo_prob
        else:
            raise NotImplementedError(self.pseudo_label_weight)

        return pseudo_label, pseudo_weight