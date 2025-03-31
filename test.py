import inspect
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TextAdapter(nn.Module):
    def __init__(self, hidden_size, adapter_start, adapter_end, adapter_dim, shared=True):
        super().__init__()
        self.adapter_start = adapter_start
        self.adapter_end = adapter_end
        self.shared = shared

        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, adapter_dim),
                nn.ReLU(),
                nn.Linear(adapter_dim, hidden_size)
            ) if adapter_start <= i <= adapter_end else None
            for i in range(12)  # Notice: 12 is the number of layers in RoBERTa-base
                                # for BERT-base/RoBERTa-base; adjust if needed
        ])

        if shared:
            self.shared_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(adapter_dim, adapter_dim),
                    nn.ReLU()
                ) if adapter_start <= i <= adapter_end else None
                for i in range(12)
            ])
        else:
            self.shared_adapters = [None] * 12

        self.scale = 1.0

    def get_adapter(self, index):
        return self.adapters[index], self.shared_adapters[index], self.scale


class TMPTTextualModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.textual_transformer_name)
        self.hidden_size = self.config.hidden_size
        self.textual_plm = AutoModel.from_pretrained(args.textual_transformer_name, self.config)

        # 插入 adapter
        self.adapter = TextAdapter(
            hidden_size=self.hidden_size,
            adapter_start=2,
            adapter_end=11,
            adapter_dim=64,
            shared=True
        )
        self.patch_model()

    def patch_model(self):
        # 给 transformer 的每层 encoder 注入 adapter_func
        for idx, layer in enumerate(self.textual_plm.encoder.layer):
            original_forward = layer.forward

            def patched_forward(module_self, hidden_states,layer_idx=idx, *args, **kwargs):
                adapter, shared_adapter, scale = self.adapter.get_adapter(layer_idx)
                residual = hidden_states
                if adapter is not None:
                    hidden_states = adapter(hidden_states)
                    if shared_adapter is not None:
                        hidden_states = shared_adapter(hidden_states)
                    hidden_states = hidden_states * scale
                    hidden_states = hidden_states + residual
                return original_forward(hidden_states, *args, **kwargs)

            layer.forward = patched_forward.__get__(layer, type(layer))

    def extract_at_mask(self, outputs, input_datas):
        if len(outputs.shape) == 2:
            return outputs
        outputs = outputs[torch.where(input_datas['text_loss_ids'] > 0)]
        outputs = outputs.view(input_datas['text_loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def forward(self, input_data):
        outputs = self.textual_plm(**{k: v for k, v in input_data.items() if k in inspect.signature(self.textual_plm.forward).parameters})
        outputs_at_mask = {k: self.extract_at_mask(v, input_data) for k, v in outputs.items()}
        return outputs_at_mask
