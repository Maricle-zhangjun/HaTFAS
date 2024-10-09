import torch
import torch.nn as nn
from torch.nn import functional as F

class DistillationLoss(nn.Module):

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module, args):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert args.distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = args.distillation_type
        self.tau = args.distillation_tau

        self.layer_ids_s = args.s_id
        self.layer_ids_t = args.t_id
        self.alpha = args.distillation_alpha
        self.beta = args.distillation_beta
        
    def forward(self, inputs, outputs, labels):

        student_outputs = outputs[0]
        class_tokens_s = outputs[1]

        # CE Loss
        base_loss = self.base_criterion(student_outputs, labels)

        if self.distillation_type == 'none':
            return base_loss 

        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs, class_tokens_t = self.teacher_model(inputs)

        # Logit-level distillation loss
        if self.distillation_type == 'soft':
            T = self.tau
            logit_distillation_loss = F.kl_div(
                F.log_softmax(student_outputs / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='batchmean',
                log_target=True
                ) * (T * T)
        elif self.distillation_type == 'hard':
            logit_distillation_loss = F.cross_entropy(student_outputs, teacher_outputs.argmax(dim=1))
        
        loss_base = (1 - self.alpha) * base_loss
        loss_logit_dist = self.alpha * logit_distillation_loss
        
        loss_fea_dist = encoder_layer_loss(class_tokens_s, class_tokens_t, self.layer_ids_s, self.layer_ids_t)
        loss_fea_dist = self.beta * loss_fea_dist
        
        return loss_base, loss_logit_dist, loss_fea_dist

# Feature-level head-aware distillation loss
def encoder_layer_loss(class_tokens_s, class_tokens_t, layer_ids_s, layer_ids_t):
    losses = []    

    for id_s, id_t in zip(layer_ids_s, layer_ids_t):
        cls_s = class_tokens_s[id_s] # (B,D_s)-->B:batch size, D_s:embedding dimension of stu (D_s=n_s×d, n_s:attn_heads of stu, d=64)
        cls_t = class_tokens_t[id_t] # (B,D_t)-->B:batch size, D_t:embedding dimension of tea (D_t=n_t×d, n_t:attn_heads of tea, d=64)
        
        cls_s = F.normalize(cls_s, dim=-1) # (B,D_s)
        cls_t = F.normalize(cls_t, dim=-1) # (B,D_t)

        corr_map = torch.mm(cls_s.transpose(0, 1), cls_t) # (D_s*D_t)

        rec_cls_s = torch.mm(cls_s,corr_map) # (B*D_t)

        t_s_diff = cls_t - rec_cls_s
        cls_loss = (t_s_diff*t_s_diff).mean()

        losses.append(cls_loss)

    enc_loss = sum(losses)/len(losses)
        
    return enc_loss

