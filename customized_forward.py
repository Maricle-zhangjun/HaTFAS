from types import MethodType
import torch

def register_forward(model, model_name):
    if model_name.split('_')[0] == 'deit':
        model.forward_features = MethodType(vit_forward_features, model)
        model.forward = MethodType(vit_forward, model)
    else:
        raise RuntimeError(f'Not defined customized method forward for model {model_name}')


def vit_forward_features(self, x, require_feat: bool = False):
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)

    class_tokens = []
    for i, blk in enumerate(self.blocks):
        norm_x = blk.norm1(x)
        attn = blk.attn(norm_x)

        class_token = attn[:, 0]
        class_tokens.append(class_token)
    
    x = self.blocks(x)
    x = self.norm(x)
    if require_feat:
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), class_tokens
        else:
            return (x[:, 0], x[:, 1]), class_tokens
    else:
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]


def vit_forward(self, x, require_feat: bool = True):

    if require_feat:
        outs = self.forward_features(x, require_feat=True)
        x = outs[0]
        class_tokens = outs[-1]
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1]) # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return (x, x_dist), class_tokens
            else:
                return (x + x_dist) / 2, class_tokens
        else:
            x = self.head(x)
        return x, class_tokens
    else:
        x = self.forward_features(x)   

        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
     
