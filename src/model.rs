use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{Bool, Device, Int},
};

use crate::{
    text_config::TextConfig,
    text_modules::{
        AttentionMask, Cache, DecoderLayer, DecoderLayerConfig, RMSNorm, RMSNormConfig,
        TextRotaryEmbedding, TextRotaryEmbeddingConfig,
    },
};

#[derive(Module, Debug)]
pub struct TextModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    rotary_emb: TextRotaryEmbedding,
}

impl TextModel {
    pub fn forward_old(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Option<Tensor<2, Bool>>,
        position_ids: Option<Tensor<3, Int>>,
        mut past_key_values: Option<&mut Cache>,
    ) -> Tensor<3> {
        let device = &input_ids.device();
        let [batch_size, seq_len] = input_ids.dims();

        let past_len = past_key_values.as_ref().map_or(0, |c| {
            c.key_states
                .iter()
                .flatten()
                .next()
                .map_or(0, |k| k.dims()[2])
        });
        let total_len = past_len + seq_len;

        let mut hidden_states = self.embed_tokens.forward(input_ids);

        let position_ids = match position_ids {
            Some(ids) => ids,
            None => {
                let ids = Tensor::arange(past_len as i64..total_len as i64, device)
                    .unsqueeze_dims::<3>(&[0, 1])
                    .expand([3, batch_size, seq_len]);
                ids
            }
        };
        let position_embeddings = self.rotary_emb.forward(position_ids);
        let linear_mask = attention_mask.clone();
        let causal_mask =
            Tensor::<2, Bool>::triu_mask([seq_len, total_len], (past_len + 1) as i64, device)
                .unsqueeze_dims::<4>(&[0, 1]); // (1, 1, seq_len, total_len)

        let full_mask = if let Some(pad_mask) = attention_mask {
            Some(pad_mask.unsqueeze_dims::<4>(&[1, 2]) | causal_mask)
        } else {
            Some(causal_mask.expand([batch_size, 1, seq_len, total_len]))
        };
        for layer in self.layers.iter() {
            // let t = std::time::Instant::now();

            let layer_mask = if layer.self_attn.is_some() {
                full_mask.clone().map(AttentionMask::Full)
            } else {
                linear_mask.clone().map(AttentionMask::Linear)
            };
            hidden_states = layer.forward(
                hidden_states,
                position_embeddings.clone(),
                layer_mask,
                past_key_values.as_deref_mut(),
            );
        }
        self.norm.forward(hidden_states)
    }
    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Option<Tensor<2, Bool>>,
        position_ids: Option<Tensor<3, Int>>,
        mut past_key_values: Option<&mut Cache>,
    ) -> Tensor<3> {
        let device = &input_ids.device();
        let [batch_size, seq_len] = input_ids.dims();

        let past_len = past_key_values.as_ref().map_or(0, |c| {
            c.key_states
                .iter()
                .flatten()
                .next()
                .map_or(0, |k| k.dims()[2])
        });
        let total_len = past_len + seq_len;

        let mut hidden_states = self.embed_tokens.forward(input_ids);

        let position_ids = match position_ids {
            Some(ids) => ids,
            None => Tensor::arange(past_len as i64..total_len as i64, device)
                .unsqueeze_dims::<3>(&[0, 1])
                .expand([3, batch_size, seq_len]),
        };
        let position_embeddings = self.rotary_emb.forward(position_ids);
        let linear_mask = attention_mask.clone();
        let causal_mask =
            Tensor::<2, Bool>::triu_mask([seq_len, total_len], (past_len + 1) as i64, device)
                .unsqueeze_dims::<4>(&[0, 1]); // (1, 1, seq_len, total_len)

        let full_mask = if let Some(pad_mask) = attention_mask {
            Some(pad_mask.unsqueeze_dims::<4>(&[1, 2]) | causal_mask)
        } else {
            Some(causal_mask.expand([batch_size, 1, seq_len, total_len]))
        };

        for layer in self.layers.iter() {
            let layer_mask = if layer.self_attn.is_some() {
                full_mask.clone().map(AttentionMask::Full)
            } else {
                linear_mask.clone().map(AttentionMask::Linear)
            };

            hidden_states = layer.forward(
                hidden_states,
                position_embeddings.clone(),
                layer_mask,
                past_key_values.as_deref_mut(),
            );
        }
        self.norm.forward(hidden_states)
    }
}

#[derive(Config, Debug)]
pub struct TextModelConfig {}

impl TextModelConfig {
    pub fn init(&self, config: &TextConfig, device: &Device) -> TextModel {
        let embed_tokens = EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
        let mut layers = vec![];
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(DecoderLayerConfig::new(layer_idx).init(config, device));
        }
        let norm = RMSNormConfig::new(config.hidden_size, config.rms_norm_eps).init(device);
        let rotary_emb = TextRotaryEmbeddingConfig::new().init(config, device);
        TextModel {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
        }
    }
}

#[derive(Module, Debug)]
pub struct TextForCausalLM {
    pub language_model: TextModel,
    pub lm_head: Linear,
}

impl TextForCausalLM {
    pub fn forward_old(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Option<Tensor<2, Bool>>,
        position_ids: Option<Tensor<3, Int>>,
        past_key_values: Option<&mut Cache>,
    ) -> Tensor<3> {
        let hidden_states =
            self.language_model
                .forward(input_ids, attention_mask, position_ids, past_key_values);

        self.lm_head.forward(hidden_states)
    }
    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Option<Tensor<2, Bool>>,
        position_ids: Option<Tensor<3, Int>>,
        past_key_values: Option<&mut Cache>,
    ) -> Tensor<3> {
        let hidden_states =
            self.language_model
                .forward(input_ids, attention_mask, position_ids, past_key_values);
        self.lm_head.forward(hidden_states)
    }
}

#[derive(Config, Debug)]
pub struct TextForCausalLMConfig {}

impl TextForCausalLMConfig {
    pub fn init(&self, config: &TextConfig, device: &Device) -> TextForCausalLM {
        let language_model = TextModelConfig::new().init(config, device);
        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);
        TextForCausalLM {
            language_model,
            lm_head,
        }
    }
}
