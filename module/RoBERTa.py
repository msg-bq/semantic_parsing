import os
import json
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer


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
        self_attn_output, self_attn_weights = self.self_attn(q, k, q, attn_mask=causal_mask)
        self_attn_output = self_attn_output.transpose(0, 1)
        x = self.norm1(decoder_input + self.dropout(self_attn_output))

        # 交叉注意力模块
        q = x.transpose(0, 1)
        k = encoder_output.transpose(0, 1)
        v = encoder_output.transpose(0, 1)
        key_padding_mask = (encoder_mask == 0) if encoder_mask is not None else None
        cross_attn_output, cross_attn_weights = self.cross_attn(q, k, v, key_padding_mask=key_padding_mask)
        cross_attn_output = cross_attn_output.transpose(0, 1)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 前馈网络模块
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = self.norm3(x + self.dropout(ff_output))

        return x, self_attn_weights, cross_attn_weights


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

        decoder_attentions = []
        cross_attentions = []

        x = decoder_input
        for layer in self.layers:
            # 获取每一层的输出和注意力权重
            layer_output, self_attn_weights, cross_attn_weights = layer(projected_encoder_output, x, encoder_mask, causal_mask)

            # 将每一层的注意力权重存储到列表中
            decoder_attentions.append(self_attn_weights)
            cross_attentions.append(cross_attn_weights)

            x = layer_output

        # 返回投影后的 encoder 输出、decoder 输出，以及注意力权重
        return projected_encoder_output, x, decoder_attentions, cross_attentions

# 完整的网络架构
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaForCausalLM, EncoderDecoderModel, EncoderDecoderConfig
from configuration_Rob import RobertaConfig

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


import torch

from transformers.modeling_outputs import Seq2SeqLMOutput

class RobertaBiLinearAttentionModel(EncoderDecoderModel):
    def __init__(self, roberta_model_name, hidden_size, num_layers, num_heads, vocab_size, epsilon=0.1,
                 feedforward_size=512, dropout=0.1):
        roberta_config = RobertaConfig.from_pretrained(roberta_model_name)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=roberta_config,
                                                                   decoder_config=roberta_config)
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
        self.loss_fn = LabelSmoothingCrossEntropy(smoothing=epsilon)

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

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    # 这里处理输入
    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            decoder_attention_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }


    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        # 如果未提供 decoder 输入，则通过标签右移生成
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        if encoder_outputs is None:
            last_hidden_state = self.encoder(input_ids, attention_mask=attention_mask,
                                           output_attentions=True, output_hidden_states=True)[0]
        else:
            last_hidden_state = encoder_outputs.last_hidden_state

        # 根据输入类型确定 decoder 输入
        if decoder_inputs_embeds is None:
            decoder_input = decoder_input_ids
        else:
            decoder_input = decoder_inputs_embeds

        if labels is not None and decoder_input is None:
            raise ValueError(
                "Decoder input is None. Make sure to provide either `decoder_input_ids` or `decoder_inputs_embeds`.")

        # 得到经过投影后的 encoder 输出和 decoder 输出（两者维度均为 1024）

        projected_encoder_output, decoder_output, decoder_attentions, cross_attentions = self.decoder(
            last_hidden_state, decoder_input, attention_mask
        )

        # 计算 Bi-linear Attention 得分（使用投影后的 encoder 表示）
        scores = self.attention(projected_encoder_output, decoder_output)
        # 将 decoder 输出映射到词汇表大小，得到 logits
        logits = self.lm_head(decoder_output)
        # 将 scores 和 logits 在最后一维进行拼接
        logits_combined = torch.cat((scores, logits), dim=-1)

        loss = None
        if labels is not None:
            logits_flat = logits_combined.view(-1, logits_combined.size(-1))
            labels_flat = labels.view(-1)
            loss = self.loss_fn(logits_flat, labels_flat)
            return (loss, scores, logits_combined)
        else:
            # 返回 Seq2SeqLMOutput，包括相关的注意力权重
            return Seq2SeqLMOutput(
                loss=loss,
                logits=logits_combined,
                past_key_values=past_key_values,  # 可以根据需要返回 past_key_values
                decoder_hidden_states=decoder_output,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                encoder_last_hidden_state=last_hidden_state,
                encoder_hidden_states=None,
                encoder_attentions=None,
            )


    # 我们在generate每次循环是加到input_ids下了，decoder是None
    # MT5是input_ids是None，通过get_encoder()直接给出来了encoder_outputs
    # def forward(
    #     self,
    #     input_ids = None,
    #     attention_mask = None,
    #     decoder_input_ids = None,
    #     decoder_attention_mask = None,
    #     head_mask = None,
    #     decoder_head_mask = None,
    #     cross_attn_head_mask = None,
    #     encoder_outputs = None,
    #     past_key_values = None,
    #     inputs_embeds = None,
    #     decoder_inputs_embeds = None,
    #     labels = None,
    #     use_cache = None,
    #     output_attentions = None,
    #     output_hidden_states = None,
    #     return_dict = None,
    # ):
    #
    #     # 如果未提供 decoder 输入，则通过标签右移生成
    #     if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
    #         decoder_input_ids = self._shift_right(labels)
    #
    #     # 如果没有提供 decoder 输入，则为 decoder_input_ids 提供一个默认的起始符号（<bos>）
    #     # 这里应该是根据decoder_input_ids来得到decoder_input_ids
    #     # 想办法手工剥离吧
    #     # 1.encoder_output只计算一次，然后保留下来
    #     # 2.在第一次计算encoder_output的那个循环令decoder_input_ids = torch.full((batch_size, 1), self.config.bos_token_id, device=input_ids.device)
    #     # 3.之后（通过判断保存的encoder_output是不是None）每次则令decoder_input_ids取input_ids的最后一个值
    #     if labels is None and self.encoder_output is None:
    #         # 默认使用一个开始符号 ID，这个值需要根据实际情况来决定
    #         # 例如：<bos> 可能是词汇表中第一个 token 的 ID
    #         batch_size = input_ids.size(0)
    #         decoder_input_ids = torch.full((batch_size, 1), self.config.bos_token_id, device=input_ids.device)
    #         self.encoder_output = self.encoder(input_ids, attention_mask=attention_mask)[0]
    #     elif labels is None:
    #         decoder_input_ids = input_ids[:, -1:]  # 使用 input_ids 的最后一个元素更新 decoder_input_ids
    #     else:
    #         encoder_output = self.encoder(input_ids, attention_mask=attention_mask)[0]
    #
    #     if self.encoder_output is not None:
    #         encoder_output = self.encoder_output
    #         # 创建一个与encoder_output前两维相匹配的attention_mask
    #         attention_mask = torch.ones(encoder_output.size(0), encoder_output.size(1), device=encoder_output.device)
    #
    #     # 根据输入类型确定 decoder 输入
    #     if decoder_inputs_embeds is None:
    #         decoder_input = decoder_input_ids
    #     else:
    #         decoder_input = decoder_inputs_embeds
    #
    #     if labels is not None and decoder_input is None:
    #         raise ValueError(
    #             "Decoder input is None. Make sure to provide either `decoder_input_ids` or `decoder_inputs_embeds`.")
    #
    #     # 得到经过投影后的 encoder 输出和 decoder 输出（两者维度均为 1024）
    #     projected_encoder_output, decoder_output, decoder_attentions, cross_attentions = self.decoder(
    #         encoder_output, decoder_input, attention_mask
    #     )
    #
    #     # 计算 Bi-linear Attention 得分（使用投影后的 encoder 表示）
    #     scores = self.attention(projected_encoder_output, decoder_output)
    #     # 将 decoder 输出映射到词汇表大小，得到 logits
    #     logits = self.lm_head(decoder_output)
    #     # 将 scores 和 logits 在最后一维进行拼接
    #     logits_combined = torch.cat((scores, logits), dim=-1)
    #
    #     loss = None
    #     if labels is not None:
    #         logits_flat = logits_combined.view(-1, logits_combined.size(-1))
    #         labels_flat = labels.view(-1)
    #         loss = self.loss_fn(logits_flat, labels_flat)
    #         return (loss, scores, logits_combined)
    #     else:
    #         # 返回 Seq2SeqLMOutput，包括相关的注意力权重
    #         return Seq2SeqLMOutput(
    #             loss=loss,
    #             logits=logits_combined,
    #             past_key_values=past_key_values,  # 可以根据需要返回 past_key_values
    #             decoder_hidden_states=decoder_output,
    #             decoder_attentions=decoder_attentions,
    #             cross_attentions=cross_attentions,
    #             encoder_last_hidden_state=encoder_output,
    #             encoder_hidden_states=None,
    #             encoder_attentions=None,
    #         )


    # def generate(self, input_ids, max_length=20, beam_width=5, temperature=1.0, top_k=50, top_p=0.9):
    #     device = input_ids.device
    #     batch_size = input_ids.size(0)
    #
    #     # 初始化 Beam Search 相关变量
    #     generated = input_ids  # [batch_size, 1]
    #     beam_scores = torch.zeros(batch_size, beam_width, device=device)  # [batch_size, beam_width]
    #     beam_seqs = generated.unsqueeze(1).repeat(1, beam_width, 1)  # [batch_size, beam_width, seq_len]
    #     active_beams = torch.ones(batch_size, beam_width, device=device, dtype=torch.bool)  # [batch_size, beam_width]
    #
    #     past_key_values = None
    #     all_beam_seqs = []  # 用于保存所有 beam 的生成结果
    #
    #     # Start Beam Search
    #     for step in range(max_length):
    #         all_beam_scores = []
    #         all_beam_seqs_step = []
    #         all_past_key_values = []
    #
    #         for beam in range(beam_width):
    #             # 每个 beam 都会独立生成
    #             beam_input_ids = beam_seqs[:, beam, :]
    #             decoder_input_ids = beam_input_ids  # 当前时刻输入是之前生成的 token
    #
    #             # 获取当前 token 的 logits
    #             logits_combined = self.forward(input_ids=beam_input_ids, decoder_input_ids=decoder_input_ids,
    #                                          past_key_values=past_key_values, use_cache=True)
    #             logits = logits_combined[:, -1, :]  # 取出最新 token 的 logits
    #
    #             logits /= temperature  # Apply temperature scaling
    #
    #             # Top-K 和 Top-P 策略
    #             if top_k > 0:
    #                 top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
    #                 logits = torch.zeros_like(logits).scatter_(-1, top_k_indices, top_k_values)
    #
    #             if top_p > 0.0:
    #                 sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    #                 cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    #                 sorted_indices_to_remove = cumulative_probs > top_p
    #                 sorted_logits[sorted_indices_to_remove] = -float('Inf')
    #                 logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    #
    #             # 计算下一个 token 的概率分布
    #             probabilities = torch.nn.functional.softmax(logits, dim=-1)  # [batch_size, vocab_size]
    #             next_token = torch.multinomial(probabilities, 1)  # [batch_size, 1]
    #
    #             # 扩展 next_token，使其与 beam_seqs 一致
    #             next_token = next_token.unsqueeze(1)  # [batch_size, 1, 1]
    #             next_token = next_token.expand(batch_size, beam_width, 1)  # [batch_size, beam_width, 1]
    #
    #             # 更新当前 beam 的 seq
    #             all_beam_seqs_step.append(torch.cat([beam_seqs, next_token], dim=-1))
    #             all_beam_scores.append(beam_scores + torch.log(probabilities.gather(1, next_token.squeeze(-1))))
    #
    #             # 更新 past_key_values
    #             if past_key_values is not None:
    #                 all_past_key_values.append(past_key_values)
    #
    #         # 汇总所有 beams 得分
    #         beam_scores = torch.stack(all_beam_scores, dim=1)  # [batch_size, beam_width, beam_width]
    #         beam_seqs = torch.stack(all_beam_seqs_step, dim=1)  # [batch_size, beam_width, seq_len + 1]
    #
    #         if all_past_key_values:
    #             past_key_values = torch.stack(all_past_key_values, dim=1)
    #         else:
    #             past_key_values = None
    #
    #         # 选择得分最高的 beams
    #         beam_scores = beam_scores.view(batch_size, -1)  # [batch_size, beam_width * beam_width]
    #         best_beam_scores, best_beam_idx = torch.topk(beam_scores, beam_width,
    #                                                      dim=-1)  # best_beam_idx: [batch_size, beam_width]
    #
    #         # 更新输出，确保 best_beam_idx 扩展成三维张量以与 beam_seqs 匹配
    #         best_beam_idx = best_beam_idx.unsqueeze(-1)  # best_beam_idx: [batch_size, beam_width, 1]
    #         best_beam_idx = best_beam_idx.expand(-1, -1, beam_seqs.size(
    #             -1))  # best_beam_idx: [batch_size, beam_width, seq_len + 1]
    #
    #         # 使用 gather 更新 beam_seqs，确保维度匹配
    #         beam_seqs = beam_seqs.gather(1, best_beam_idx)  # beam_seqs: [batch_size, beam_width, seq_len + 1]
    #
    #         if step == max_length - 1:
    #             all_beam_seqs = beam_seqs  # 记录最终生成的序列
    #
    #     # 返回最终生成的多个序列
    #     return all_beam_seqs

    def _update_past_key_values(self, past_key_values):
        """
        更新缓存的 key/value 以支持更高效的生成。
        """
        return past_key_values  # 目前没有实现缓存更新逻辑，根据实际需求修改

    def _resize_token_embeddings(self, new_num_tokens):
        # Resize input embeddings
        old_embeddings = self.encoder.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.encoder.embeddings.word_embeddings = new_embeddings

        self.lm_head = nn.Linear(self.hidden_size, new_num_tokens - 64, bias=False)

        return self.encoder.embeddings.word_embeddings

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens):
        # Create new embeddings tensor with updated number of tokens
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
        # Copy old embeddings to new one
        new_embeddings.weight.data[:old_embeddings.weight.size(0), :] = old_embeddings.weight.data
        return new_embeddings