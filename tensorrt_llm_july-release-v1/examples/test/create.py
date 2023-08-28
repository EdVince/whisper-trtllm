import math
import torch
import torch.nn as nn
from activations import ACT2FN

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


class SimpleConvTorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WhisperEncoder()

    def forward(self, x):
        x = self.encoder(x)
        return x

if __name__ == '__main__':

    torch_net = SimpleConvTorchNet()
    torch.save(torch_net.state_dict(),'weight.pth')

    input = torch.rand(1,80,3000)
    output = torch_net(input)
    print(output.shape)