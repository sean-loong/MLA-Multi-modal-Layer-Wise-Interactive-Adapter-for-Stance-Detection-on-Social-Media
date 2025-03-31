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