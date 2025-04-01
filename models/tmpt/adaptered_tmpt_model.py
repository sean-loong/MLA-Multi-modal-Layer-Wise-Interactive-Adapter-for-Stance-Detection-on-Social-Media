import sys
sys.path.append('./')
from collections import OrderedDict
import torch
import torch.nn as nn
from models.tmpt.adaptered_tmpt_textual_model import TMPTTextualModel
from models.tmpt.adaptered_tmpt_visual_model import TMPTVisualModel
from models.tmpt.adaptered_tmpt_textual_model import adaptered_TMPTTextualModel
from models.tmpt.adaptered_tmpt_visual_model import adaptered_TMPTVisualModel


class AdapterLearner(nn.Module):
    # 构建文本和图像两路 adapter（按层插入）
    # 实现 adapter_func，用于在特定层插入 adapter 模块
    
    def __init__(self, args, tmpt_model):
        super().__init__()
        # 提取编码器
        text_enc = tmpt_model.tmpt_textual_model
        vis_enc = tmpt_model.tmpt_visual_model
        
        # build multi-modal adapter
        
        self.text_adapter = self._build_adapter(
            # 参数分别为：d_model, n_layers, l_start, l_end, mid_dim, dtype
            d_model=text_enc.d_model,
            n_layers=text_enc.n_layers,
            l_start=args.ADAPTER_START,
            l_end = args.ADAPTER_END,
            mid_dim = args.ADAPTER_DIM,
            dtype = tmpt_model.dtype
        )
        
        self.visual_adapter = self._build_adapter(
            d_model=vis_enc.d_model,
            n_layers=vis_enc.n_layers,
            l_start=args.ADAPTER_START,
            l_end = args.ADAPTER_END,
            mid_dim = args.ADAPTER_DIM,
            dtype = tmpt_model.dtype
        )

        self.shared_adapter = self._build_adapter(
            args.ADAPTER_DIM,
            n_layers=vis_enc.n_layers, 
            l_start=args.ADAPTER_START,
            l_end = args.ADAPTER_END,
            mid_dim = args.ADAPTER_DIM,
            dtype = tmpt_model.dtype
        )
        
        self.adapter_scale = float(args.ADAPTER_SCALE)
        self.text_adapter_func = lambda x: self.return_text_adapter(index=x)
        self.visual_adapter_func = lambda x: self.return_visual_adapter(index=x)
        
        
    def return_text_adapter(self, index):
        return self.text_adapter[index], self.shared_adapter[index], self.adapter_scale

    def return_visual_adapter(self, index):
        return self.visual_adapter[index], self.shared_adapter[index], self.adapter_scale

    def _build_adapter(self, d_model, n_layers, l_start, l_end, mid_dim, dtype):
        adapter = [None] * (n_layers + 1)
        for i in range(l_start, l_end+1):
            if mid_dim == d_model:
                adapter[i] = nn.Sequential(
                    nn.Linear(d_model, mid_dim),
                    nn.ReLU()
                )
            else:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("down", nn.Sequential(nn.Linear(d_model, mid_dim), nn.ReLU())),
                    ("up", nn.Linear(mid_dim, d_model))
                ]))
        adapter = nn.ModuleList([a for a in adapter])
        for m in adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        if dtype == torch.float16:
            for m in adapter.modules():
                m.half()
        return adapter

    def forward(self):
        return self.text_adapter_func, self.visual_adapter_func



class adaptered_TMPTModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tmpt_textual_model = adaptered_TMPTTextualModel(args)
        self.tmpt_visual_model = adaptered_TMPTVisualModel(args)
        self.adapter_learner = AdapterLearner(args, self)
        # 这里的 self.adapter_learner 是一个 AdapterLearner 对象，包含了文本和视觉两路的 adapter
        # 以及一个共享的 adapter。它的作用是为每一层的 transformer 模型添加 adapter 模块。
        
        
        
        for k, p in self.tmpt_visual_model.named_parameters():
            if "prompt" not in k and "pooler" not in k:
                p.requires_grad = False

        if args.linear_injection == -1:
            linear_injection = min(self.tmpt_textual_model.hidden_size, self.tmpt_visual_model.hidden_size)
        else:
            linear_injection = args.linear_injection

        self.textual_transformer_linear = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.tmpt_textual_model.hidden_size, linear_injection)),
            ('layerNorm', nn.BatchNorm1d(linear_injection)),
            ('activate', nn.LeakyReLU(0.2))
        ]))

        self.visual_transformer_linear = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.tmpt_visual_model.hidden_size, linear_injection)),
            ('layerNorm', nn.BatchNorm1d(linear_injection)),
            ('activate', nn.LeakyReLU(0.2))
        ]))

        self.classifier = nn.Linear(linear_injection*2, args.label_size)
        
    @property
    def dtype(self):
        return next(self.tmpt_visual_model.parameters()).dtype
    # 

    def forward(self, input_data):
        text_adapter_func, visual_adapter_func = self.adapter_learner.forward()
        # 调用了 self.adapter_learner()，但 AdapterLearner 是个 nn.Module 类，没有 __call__ 方法支持这个调用行为。
        
        textual_outputs = self.tmpt_textual_model(input_data, text_adapter_func)
        # textual_outputs = self.tmpt_textual_model(input_data)
        text_pooled_output = self.textual_transformer_linear(textual_outputs['last_hidden_state'])

        visual_outputs = self.tmpt_visual_model(input_data, visual_adapter_func)
        # visual_outputs = self.tmpt_visual_model(input_data)
        image_pooled_output = self.visual_transformer_linear(visual_outputs['last_hidden_state'])

        logits = self.classifier(torch.cat([text_pooled_output, image_pooled_output], dim=-1))
        return logits


if __name__ == '__main__':
    class Args():
        label_size = 2
        linear_injection = -1
        prompt_dropout = 0.2
        visual_soft_tokens = 5
        visual_soft_prompt_dropout = 0.2
        # model config
        textual_transformer_tokenizer_name = 'model_state/vinai/bertweet-base'
        textual_transformer_name = 'model_state/vinai/bertweet-base'

        visual_transformer_tokenizer_name = 'model_state/google/vit-base-patch16-224'
        visual_transformer_name = 'model_state/google/vit-base-patch16-224'

    import torch
    args = Args()
    model = TMPTModel(args)
    text_ids = torch.randint(low=0, high=64001, size=[16, 128], dtype=torch.long)
    text_masks = torch.ones(size=[16, 128], dtype=torch.long)
    text_loss_ids = torch.zeros(size=[16, 512], dtype=torch.long)
    text_loss_ids[:, 100] = 1
    image_tensor = torch.randn(size=[16, 3, 224, 224])
    input_data = {'input_ids': text_ids, 'attention_mask': text_masks, 'text_loss_ids': text_loss_ids, 'pixel_values': image_tensor}
    logits = model(input_data)
    print(f'logits.shape: {logits.shape}')