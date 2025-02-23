import os
import json
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaConfig


# 修改 DecoderLayer，使得隐藏层维度为 1024
class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, feedforward_size, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_size, feedforward_size)
        self.linear2 = nn.Linear(feedforward_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_output, decoder_input, encoder_mask=None, causal_mask=None):
        # 自注意力模块
        x = decoder_input
        q = k = x.transpose(0, 1)
        self_attn_output, _ = self.self_attn(q, k, q, attn_mask=causal_mask)
        self_attn_output = self_attn_output.transpose(0, 1)
        x = self.norm1(decoder_input + self.dropout(self_attn_output))

        # 交叉注意力模块
        q = x.transpose(0, 1)
        k = encoder_output.transpose(0, 1)
        v = encoder_output.transpose(0, 1)
        key_padding_mask = (encoder_mask == 0) if encoder_mask is not None else None
        cross_attn_output, _ = self.cross_attn(q, k, v, key_padding_mask=key_padding_mask)
        cross_attn_output = cross_attn_output.transpose(0, 1)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 前馈网络模块
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = self.norm3(x + self.dropout(ff_output))
        return x


# 修改 BiLinear Attention
# 这个我目前还没想好怎么用
# 改一改？
class BiLinearAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BiLinearAttention, self).__init__()
        self.attn = nn.Bilinear(hidden_size, hidden_size, 1)

    def forward(self, encoder_output, decoder_output):
        B, seq_len_dec, H = decoder_output.shape
        B, seq_len_enc, H = encoder_output.shape
        # 扩展后维度都应该是 hidden_size (1024)
        decoder_expanded = decoder_output.unsqueeze(2).expand(B, seq_len_dec, seq_len_enc, H)
        encoder_expanded = encoder_output.unsqueeze(1).expand(B, seq_len_dec, seq_len_enc, H)
        scores = self.attn(encoder_expanded.reshape(-1, H), decoder_expanded.reshape(-1, H))
        scores = scores.view(B, seq_len_dec, seq_len_enc)
        return scores


# 修改 Decoder，返回投影后的 encoder 输出以及 decoder 输出
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_heads, feedforward_size, dropout=0.1, encoder=None):
        """
        input_dim: encoder 的输出维度 (768)
        hidden_size: decoder 隐藏层维度 (1024)
        encoder_embeddings: 共享的 encoder 嵌入层
        """
        super(Decoder, self).__init__()
        self.encoder_projection = nn.Linear(input_dim, hidden_size)
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self._encoder = encoder  # 保存 encoder 的引用

        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, feedforward_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.hidden_size = hidden_size

    @property
    def embedding(self):
        # 每次使用时动态获取 encoder 的 embedding
        return self._encoder.embeddings.word_embeddings

    def forward(self, encoder_output, decoder_input, encoder_mask=None):
        # 对 encoder_output 进行投影，将维度从 768 转为 1024
        projected_encoder_output = self.encoder_projection(encoder_output)

        # 根据输入类型处理 decoder_input
        if decoder_input.dtype in (torch.int64, torch.long):
            decoder_input = self.embedding(decoder_input)
            decoder_input = self.input_projection(decoder_input)
        else:
            decoder_input = self.input_projection(decoder_input)

        seq_len = decoder_input.size(1)
        # 创建 causal mask，用于自注意力（防止信息泄露）
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=decoder_input.device), diagonal=1).bool()

        x = decoder_input
        for layer in self.layers:
            x = layer(projected_encoder_output, x, encoder_mask, causal_mask)
        # 返回投影后的 encoder 输出和 decoder 的输出
        return projected_encoder_output, x


# 完整的网络架构
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig, RobertaForCausalLM, PreTrainedModel


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class RobertaBiLinearAttentionModel(RobertaForCausalLM):
    def __init__(self, roberta_model_name, hidden_size, num_layers, num_heads, vocab_size, epsilon=0.1,
                 feedforward_size=512, dropout=0.1):
        config = RobertaConfig.from_pretrained(roberta_model_name)

        super(RobertaBiLinearAttentionModel, self).__init__(config)

        # 使用预训练的 RoBERTa 作为 Encoder
        self.encoder = RobertaModel.from_pretrained(roberta_model_name)
        encoder_hidden_size = self.encoder.config.hidden_size  # 768
        self.decoder = Decoder(
            input_dim=encoder_hidden_size,
            hidden_size=hidden_size,  # 1024
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            encoder=self.encoder  # 将 encoder 实例传给 decoder
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.feedforward_size = feedforward_size
        self.epsilon = epsilon
        self.dropout = dropout

        self.attention = BiLinearAttention(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.epsilon = epsilon  # 用于调整损失
        # self.loss_fn = nn.CrossEntropyLoss(label_smoothing = epsilon)
        self.loss_fn = LabelSmoothingCrossEntropy(smoothing = epsilon)

        # 保存 Encoder 配置
        self.config = self.encoder.config

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.bos_token_id
        pad_token_id = self.config.pad_token_id

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def forward(
            self,
            input_ids,
            decoder_input_ids=None,
            decoder_inputs_embeds=None,
            attention_mask=None,
            labels=None,
            past_key_values=None,  # 新增，兼容 generate
            use_cache=False,  # 新增，用于控制是否返回缓存
            **kwargs  # 捕获其它额外的关键字参数
    ):
        """
        Args:
            input_ids: 编码器输入
            decoder_input_ids: 解码器输入id
            decoder_inputs_embeds: 解码器输入embeds
            attention_mask: 注意力 mask
            labels: 标签（用于训练时计算 loss）
            past_key_values: 缓存的 key/value（目前未使用，可以后续扩展支持缓存）
            use_cache: 是否返回缓存（目前未实现缓存逻辑）
            kwargs: 其它额外参数
        """
        # 编码器输出，形状为 [batch_size, seq_len_enc, 768]
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)[0]

        # 如果未提供 decoder 输入，则通过标签右移生成
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        # 根据输入类型确定 decoder 输入
        if decoder_inputs_embeds is None:
            decoder_input = decoder_input_ids
        else:
            decoder_input = decoder_inputs_embeds

        # 得到经过投影后的 encoder 输出和 decoder 输出（两者维度均为 1024）
        projected_encoder_output, decoder_output = self.decoder(encoder_output, decoder_input, attention_mask)

        # 计算 Bi-linear Attention 得分（使用投影后的 encoder 表示）
        scores = self.attention(projected_encoder_output, decoder_output)
        # 将 decoder 输出映射到词汇表大小，得到 logits
        logits = self.lm_head(decoder_output)
        # 将 scores 和 logits 在最后一维进行拼接
        logits_combined = torch.cat((scores, logits), dim=-1)

        if labels is not None:
            logits_flat = logits_combined.view(-1, logits_combined.size(-1))
            labels_flat = labels.view(-1)
            loss = self.loss_fn(logits_flat, labels_flat)
            return (loss, scores, logits_combined)
        else:
            return logits_combined


    def _resize_token_embeddings(self, new_num_tokens):
        # Resize input embeddings
        old_embeddings = self.encoder.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.encoder.embeddings.word_embeddings = new_embeddings

        return self.encoder.embeddings.word_embeddings

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens):
        # Create new embeddings tensor with updated number of tokens
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
        # Copy old embeddings to new one
        new_embeddings.weight.data[:old_embeddings.weight.size(0), :] = old_embeddings.weight.data
        return new_embeddings

    # def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50):
    #     """
    #     生成文本的函数
    #     :param input_ids: 输入的 token ids，形状为 [B, seq_len]
    #     :param max_length: 生成的最大长度
    #     :param temperature: 控制生成随机性的温度（越高越随机，越低越确定）
    #     :param top_k: 用于生成时限制候选的大小，避免生成概率较低的 token
    #     :return: 生成的 token ids
    #     """
    #     device = input_ids.device
    #     generated_ids = input_ids.clone()  # 克隆输入的 id，用于生成新的序列
    #
    #     for _ in range(max_length):
    #         # 获取当前的 decoder 输入
    #         decoder_input_ids = generated_ids[:, -1:]  # 取当前序列的最后一个 token 作为输入
    #
    #         # 获取模型的输出
    #         logits = self.forward(input_ids, decoder_input_ids=decoder_input_ids)[0]  # 获取预测的 logits
    #         logits = logits[:, -1, :]  # 取出最后一个时间步的 logits，形状为 [B, vocab_size]
    #
    #         # 使用 temperature 调整 logits
    #         logits = logits / temperature
    #
    #         # 使用 top_k 限制候选 token
    #         if top_k > 0:
    #             values, indices = torch.topk(logits, top_k)
    #             logits = torch.full_like(logits, -float('Inf'))
    #             logits.scatter_(1, indices, values)
    #
    #         # 使用 softmax 转化为概率
    #         probs = torch.nn.functional.softmax(logits, dim=-1)
    #
    #         # 采样生成下一个 token
    #         next_token = torch.multinomial(probs, 1)  # 从概率分布中采样一个 token
    #
    #         # 将生成的 token 添加到序列中
    #         generated_ids = torch.cat((generated_ids, next_token), dim=1)
    #
    #         # 如果生成了 EOS token，则停止生成
    #         if next_token.item() == self.config.eos_token_id:
    #             break
    #
    #     return generated_ids
    #
    # def save_pretrained(self, save_directory):
    #     """
    #     Saves the model to a specified directory, including the model weights and configuration.
    #     """
    #     if not os.path.exists(save_directory):
    #         os.makedirs(save_directory)
    #
    #     # 保存模型的状态字典
    #     model_path = os.path.join(save_directory, "pytorch_model.bin")
    #     torch.save(self.state_dict(), model_path)
    #
    #     # 保存配置文件
    #     config = {
    #         "hidden_size": self.hidden_size,
    #         "num_layers": self.num_layers,
    #         "num_heads": self.num_heads,
    #         "vocab_size": self.vocab_size,
    #         "feedforward_size": self.feedforward_size,
    #         "epsilon": self.epsilon,
    #         "dropout": self.dropout,
    #     }
    #     config_path = os.path.join(save_directory, "config.json")
    #     with open(config_path, 'w') as config_file:
    #         json.dump(config, config_file)
    #
    #     print(f"模型已保存到：{save_directory}")
    #
    # @classmethod
    # def from_pretrained(cls, save_directory):
    #     """
    #     从指定的目录加载模型及其配置
    #     """
    #     # 加载配置
    #     config_path = os.path.join(save_directory, "config.json")
    #     with open(config_path, 'r') as config_file:
    #         config = json.load(config_file)
    #
    #     # 创建模型
    #     model = cls(
    #         roberta_model_name="roberta-base",  # 你也可以加载模型名称，或者通过配置获取
    #         hidden_size=config["hidden_size"],
    #         num_layers=config["num_layers"],
    #         num_heads=config["num_heads"],
    #         vocab_size=config["vocab_size"],
    #         feedforward_size=config["feedforward_size"],
    #         epsilon=config["epsilon"],
    #         dropout=config["dropout"]
    #     )
    #
    #     # 加载权重
    #     model_path = os.path.join(save_directory, "pytorch_model.bin")
    #     model.load_state_dict(torch.load(model_path))
    #
    #     return model