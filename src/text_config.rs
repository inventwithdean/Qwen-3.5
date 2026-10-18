use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize)]
pub struct TextRopeParams {
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],
    pub rope_type: String,
    pub rope_theta: usize,
    pub partial_rotary_factor: f32,
}

// #[derive(Serialize, Deserialize)]
// pub enum LayerType {
//     LinearAttention,
//     FullAttention,
// }

#[derive(Serialize, Deserialize)]
pub struct TextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub attn_output_gate: bool,
    pub dtype: String,
    pub eos_token_id: usize,
    pub full_attention_interval: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub max_position_embeddings: usize,
    pub mlp_only_layers: Vec<usize>,
    pub model_type: String,
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: usize,
    pub mamba_ssm_dtype: String,
    pub rope_parameters: TextRopeParams,
}

/// Returns text config of Qwen 3.5 2B model
pub fn get_text_config() -> TextConfig {
    serde_json::from_value(json!({
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attn_output_gate": true,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 6144,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention"
        ],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "num_attention_heads": 8,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": true,
        "use_cache": true,
        "vocab_size": 248320,
        "mamba_ssm_dtype": "float32",
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [
                11,
                11,
                10
            ],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25
        }
    }))
    .unwrap()
}
