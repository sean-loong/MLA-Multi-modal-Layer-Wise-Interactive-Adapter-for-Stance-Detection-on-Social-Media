import inspect
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TMPTTextualModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.config = AutoConfig.from_pretrained(args.textual_transformer_name)
        self.hidden_size = self.config.hidden_size
        self.textual_plm = AutoModel.from_pretrained(args.textual_transformer_name, self.config)

    def extract_at_mask(self, outputs, input_datas):
        if len(outputs.shape) == 2:
            return outputs
        outputs = outputs[torch.where(input_datas['text_loss_ids']>0)]
        outputs = outputs.view(input_datas['text_loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            # 如果每个样本只提取了 一个 token（也就是 outputs.shape[1] == 1），那当前 shape 是 [B, 1, H]，我们通常希望变成 [B, H]：
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def forward(self, input_data):
        outputs = self.textual_plm(**{k: v for k, v in input_data.items() if k in inspect.signature(self.textual_plm.forward).parameters})
        # 提取 transformer 模型 forward() 函数中可接受的参数名。
        outputs_at_mask = {k: self.extract_at_mask(v, input_data) for k, v in outputs.items()}
        return outputs_at_mask
    #         {
        #     'last_hidden_state': [B, H] 或 [B, M, H],
        #     'pooler_output': [B, H]
        #     ...
    #         }
    @property
    def d_model(self):
        return self.hidden_size

    @property
    def n_layers(self):
        return self.config.num_hidden_layers

class adaptered_TMPTTextualModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.textual_transformer_name)
        self.hidden_size = self.config.hidden_size
        self.textual_plm = AutoModel.from_pretrained(args.textual_transformer_name, self.config)

    @property
    def d_model(self):
        return self.hidden_size

    @property
    def n_layers(self):
        return self.config.num_hidden_layers

    # @property
    # def dtype(self):
    #     return next(self.textual_plm.parameters()).dtype

    def extract_at_mask(self, outputs, input_datas):
        if len(outputs.shape) == 2:
            return outputs
        outputs = outputs[torch.where(input_datas['text_loss_ids'] > 0)]
        outputs = outputs.view(input_datas['text_loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def prepare_attention_mask(self, attention_mask):
        # 转成 [B, 1, 1, L] 并做 mask 填充
        extended = attention_mask[:, None, None, :]  # [B, 1, 1, L]
        extended = (1.0 - extended.float()) * -10000.0
        return extended

    
    def forward(self, input_data, adapter_func=None):
        """
        如果 adapter_func 被传入，则对 transformer 的每一层进行 adapter 注入
        """

        # 如果没有 adapter，正常前向传播
        if adapter_func is None:
            outputs = self.textual_plm(**{k: v for k, v in input_data.items() if k in inspect.signature(self.textual_plm.forward).parameters})
        else:
            # 复制 config 并替换 encoder 的每一层 forward 逻辑
            # 这里只处理 encoder 部分
            hidden_states = self.textual_plm.embeddings(input_data["input_ids"])
            attention_mask = input_data.get("attention_mask", None)
            extended_attention_mask = self.prepare_attention_mask(attention_mask)

            for i, layer_module in enumerate(self.textual_plm.encoder.layer):
                layer_adapter, shared_adapter, scale = adapter_func(i + 1)  # 注意：层数从 1 开始
                if layer_adapter is not None:
                    residual = hidden_states
                    hidden_states = layer_module(hidden_states, attention_mask=extended_attention_mask)[0]
                    y = layer_adapter.down(residual)
                    y = shared_adapter(y)
                    y = layer_adapter.up(y)
                    hidden_states = hidden_states + scale * y
                else:
                    hidden_states = layer_module(hidden_states, attention_mask=extended_attention_mask)[0]

            sequence_output = hidden_states
            pooled_output = self.textual_plm.pooler(sequence_output) if self.textual_plm.pooler is not None else None

            outputs = {
                "last_hidden_state": sequence_output,
                "pooler_output": pooled_output
            }

        outputs_at_mask = {k: self.extract_at_mask(v, input_data) for k, v in outputs.items()}
        return outputs_at_mask

if __name__ == '__main__':
    class Args():
        # model config
        textual_transformer_tokenizer_name = 'model_state/roberta-base'
        textual_transformer_name = 'model_state/roberta-base'

    import torch
    args = Args()
    model = TMPTTextualModel(args)
    text_ids = torch.randint(low=0, high=50264, size=[16, 512], dtype=torch.long)
    text_masks = torch.ones(size=[16, 512], dtype=torch.long)
    # 额外构造的一个 掩码矩阵，用来指示“想从 transformer 输出中提取哪些 token 的表示
    text_loss_ids = torch.zeros(size=[16, 512], dtype=torch.long)
    text_loss_ids[:, 100] = 1
    input_data = {'input_ids': text_ids, 'attention_mask': text_masks, 'text_loss_ids': text_loss_ids}
    logits = model(input_data)
    print(f'logits.shape: {logits.shape}')