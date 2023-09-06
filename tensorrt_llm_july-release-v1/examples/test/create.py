import math
import torch
import torch.nn as nn
from activations import ACT2FN
from typing import Optional, Tuple

class WhisperEncoderAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        
        # bsz=1, tgt_len=1500, _取决于模型大小
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output

class WhisperEncoderLayer(nn.Module):
    def __init__(self, d_model=512, encoder_attention_heads=8, activation_function='gelu', encoder_ffn_dim=2048):
        super().__init__()
        self.embed_dim = d_model
        self.self_attn = WhisperEncoderAttention(
            embed_dim=self.embed_dim,
            num_heads=encoder_attention_heads,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = ACT2FN[activation_function]
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states

        return outputs

class WhisperEncoder(nn.Module):
    def __init__(self, 
                 d_model=512, num_mel_bins=80, max_source_positions=1500,
                 encoder_layers=6,
                 encoder_attention_heads=8, activation_function='gelu', encoder_ffn_dim=2048):
        super().__init__()

        embed_dim = d_model

        self.conv1 = nn.Conv1d(num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(max_source_positions, embed_dim)

        self.layers = nn.ModuleList([WhisperEncoderLayer(d_model=d_model, 
                                                         encoder_attention_heads=encoder_attention_heads, 
                                                         activation_function=activation_function, 
                                                         encoder_ffn_dim=encoder_ffn_dim) 
                    for _ in range(encoder_layers)])
        
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_features, # (1,80,3000)
    ):

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        
        embed_pos = self.embed_positions.weight
        hidden_states = inputs_embeds + embed_pos

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class WhisperDecoderAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # Copied from transformers.models.bart.modeling_bart.BartAttention._shape with BART->whisper
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # Copied from transformers.models.bart.modeling_bart.BartAttention.forward with BART->whisper
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):

        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling

        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, past_key_value

class WhisperDecoderLayer(nn.Module):
    def __init__(self, d_model = 512, decoder_attention_heads = 8, activation_function = 'gelu', decoder_ffn_dim = 2048):
        super().__init__()
        self.embed_dim = d_model

        self.self_attn = WhisperDecoderAttention(embed_dim=self.embed_dim,num_heads=decoder_attention_heads)
        self.activation_fn = ACT2FN[activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperDecoderAttention(self.embed_dim,decoder_attention_heads,)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, decoder_ffn_dim)
        self.fc2 = nn.Linear(decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            key_value_states=None,
            past_key_value=self_attn_past_key_value
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        hidden_states, cross_attn_present_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            past_key_value=cross_attn_past_key_value,
        )
        hidden_states = residual + hidden_states

        # add cross-attn to positions 3,4 of present_key_value tuple
        present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        outputs += (present_key_value,)

        return outputs

class SimpleConvTorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = WhisperDecoderAttention()

    def forward(self, hidden_states,key_value_states,past_key_value):
        attn_output, past_key_value = self.attn(hidden_states,key_value_states,past_key_value)
        return attn_output, past_key_value

if __name__ == '__main__':

    torch_net = SimpleConvTorchNet()
    torch.save(torch_net.state_dict(),'weight.pth')
    
    attn, past = torch_net(torch.rand(1,1,512),torch.rand(1,1500,512),(torch.rand(1,8,1500,64),torch.rand(1,8,1500,64)))
    print(attn.shape,past[0].shape,past[1].shape)
    
    attn, past = torch_net(torch.rand(1,1,512),torch.rand(1,1500,512),None)
    print(attn.shape,past[0].shape,past[1].shape)