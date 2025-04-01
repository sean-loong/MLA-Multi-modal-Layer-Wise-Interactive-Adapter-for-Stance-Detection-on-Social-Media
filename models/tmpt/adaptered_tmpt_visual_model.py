import math
import inspect
import torch
import torch.nn as nn
from torch.nn import Dropout
from transformers import AutoConfig, AutoModel


class TMPTVisualModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.config = AutoConfig.from_pretrained(args.visual_transformer_name)
        self.visual_plm = AutoModel.from_pretrained(args.visual_transformer_name, self.config)
        
        
        self.visual_soft_tokens = args.visual_soft_tokens
        # 要插入的 soft prompt token 数量
        self.hidden_size = self.config.hidden_size
        # 视觉模型每个 token 的向量维度（如 768）
        self.soft_prompt_dropout = Dropout(args.visual_soft_prompt_dropout)
        # dropout 防止 prompt overfit
        
        val = math.sqrt(6. / float(self.hidden_size * 2))
        self.soft_prompt_embeds = nn.Parameter(torch.zeros(1, self.visual_soft_tokens, self.hidden_size))
        # xavier_uniform initialization
        nn.init.uniform_(self.soft_prompt_embeds.data, -val, val)

    def incorporate_prompt(self, pixel_values):
        # combine prompt embeddings with text embeddings
        batch_size = pixel_values.shape[0] # (batch_size, 3, 224, 224)
        x = self.visual_plm.embeddings(pixel_values)  # x.shape == [batch_size, L, hidden_size]
        # L = 1 + N_patch，第一个 token 是 [CLS]
        x = torch.cat((
                x[:, :1, :], # CLS token: shape [B, 1, H]
                self.soft_prompt_dropout(self.soft_prompt_embeds.expand(batch_size, -1, -1)),
                x[:, 1:, :]  # patch tokens: shape [B, L-1, H]
            ), dim=1)
        
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.visual_plm.encoder.eval()
            self.visual_plm.embeddings.eval()
            self.visual_plm.layernorm.eval()
            self.visual_plm.pooler.eval()
            self.soft_prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward(self, input_data):
        embedding_output = self.incorporate_prompt(**{k: v for k, v in input_data.items() if k in inspect.signature(self.incorporate_prompt).parameters})
        encoder_outputs = self.visual_plm.encoder(embedding_output)

        last_hidden_state = encoder_outputs['last_hidden_state']
        pooled_output = self.visual_plm.pooler(last_hidden_state)
        soft_hidden_state = self.visual_plm.layernorm(last_hidden_state)[:, 1:1+self.visual_soft_tokens, :]
        soft_hidden_state = torch.avg_pool1d(soft_hidden_state.transpose(1, 2), kernel_size=self.visual_soft_tokens).squeeze(-1)
        return {
            'last_hidden_state': soft_hidden_state,
            'pooler_output': pooled_output
        }
    @property
    def d_model(self):
        return self.hidden_size

    @property
    def n_layers(self):
        return self.config.num_hidden_layers

class adaptered_TMPTVisualModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.config = AutoConfig.from_pretrained(args.visual_transformer_name)
        self.visual_plm = AutoModel.from_pretrained(args.visual_transformer_name, self.config)

        self.visual_soft_tokens = args.visual_soft_tokens
        self.hidden_size = self.config.hidden_size
        self.soft_prompt_dropout = Dropout(args.visual_soft_prompt_dropout)

        val = math.sqrt(6. / float(self.hidden_size * 2))
        self.soft_prompt_embeds = nn.Parameter(torch.zeros(1, self.visual_soft_tokens, self.hidden_size))
        nn.init.uniform_(self.soft_prompt_embeds.data, -val, val)

    @property
    def d_model(self):
        return self.hidden_size

    @property
    def n_layers(self):
        return self.config.num_hidden_layers

    @property
    def dtype(self):
        return next(self.visual_plm.parameters()).dtype

    def incorporate_prompt(self, pixel_values):
        batch_size = pixel_values.shape[0]
        x = self.visual_plm.embeddings(pixel_values)
        x = torch.cat((
            x[:, :1, :],
            self.soft_prompt_dropout(self.soft_prompt_embeds.expand(batch_size, -1, -1)),
            x[:, 1:, :]
        ), dim=1)
        return x

    def train(self, mode=True):
        if mode:
            self.visual_plm.encoder.eval()
            self.visual_plm.embeddings.eval()
            self.visual_plm.layernorm.eval()
            self.visual_plm.pooler.eval()
            self.soft_prompt_dropout.train()
        else:
            for module in self.children():
                module.train(mode)

    def forward(self, input_data, adapter_func=None):
        embedding_output = self.incorporate_prompt(**{
            k: v for k, v in input_data.items()
            if k in inspect.signature(self.incorporate_prompt).parameters
        })

        hidden_states = embedding_output
        attention_mask = None  # ViT 没有传统的 attention_mask，可选

        if adapter_func is None:
            encoder_outputs = self.visual_plm.encoder(hidden_states)
            hidden_states = encoder_outputs['last_hidden_state']
        else:
            for i, layer_module in enumerate(self.visual_plm.encoder.layer):
                layer_adapter, shared_adapter, scale = adapter_func(i + 1)
                if layer_adapter is not None:
                    residual = hidden_states
                    hidden_states = layer_module(hidden_states)[0]
                    y = layer_adapter.down(residual)
                    y = shared_adapter(y)
                    y = layer_adapter.up(y)
                    hidden_states = hidden_states + scale * y
                else:
                    hidden_states = layer_module(hidden_states)[0]

        pooled_output = self.visual_plm.pooler(hidden_states)
        soft_hidden_state = self.visual_plm.layernorm(hidden_states)[:, 1:1 + self.visual_soft_tokens, :]
        soft_hidden_state = torch.avg_pool1d(
            soft_hidden_state.transpose(1, 2),
            kernel_size=self.visual_soft_tokens
        ).squeeze(-1)

        return {
            'last_hidden_state': soft_hidden_state,
            'pooler_output': pooled_output
        }
        

if __name__ == '__main__':
    class Args():
        # model config
        visual_transformer_name = 'model_state/google/vit-base-patch16-224'
        visual_soft_tokens = 5
        visual_soft_prompt_dropout = 0.2

    import torch
    args = Args()
    model = TMPTVisualModel(args)
    image_tensor = torch.randn(size=[16, 3, 224, 224])
    input_data = {'pixel_values': image_tensor}
    logits = model(input_data)
    print(f'logits.shape: {logits.shape}')