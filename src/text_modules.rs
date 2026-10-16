use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    tensor::{Bool, Device, Int, activation::silu, module::conv1d, ops::ConvOptions, s},
};

use crate::text_config::TextConfig;

#[derive(Module, Debug)]
pub struct TextRotaryEmbedding {
    inv_freq: Tensor<1>, // (dim / 2,)
    mrope_section: [usize; 3],
}

impl TextRotaryEmbedding {
    pub fn forward(&self, position_ids: Tensor<3, Int>) -> (Tensor<3>, Tensor<3>) {
        // position_ids: (3, batch_size, seq_len)
        let position_ids = position_ids.unsqueeze_dim::<4>(3); // (3, batch_size, seq_len, 1)
        let inv_freqs = self.inv_freq.clone().unsqueeze_dims::<4>(&[0, 1, 2]); // (1, 1, 1, dim / 2)

        let freqs = position_ids.float() * inv_freqs; // (3, batch_size, seq_len, dim / 2)
        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        let cos = self.recomposition_frequencies(cos);
        let sin = self.recomposition_frequencies(sin);

        (cos, sin)
    }

    fn recomposition_frequencies(&self, freq: Tensor<4>) -> Tensor<3> {
        // freq: (3, batch_size, seq_len, dim / 2)
        let mut freqs_thw = freq.clone().slice(s![0..1, .., .., ..]).squeeze_dim::<3>(0); // (batch_size, seq_len, dim / 2)
        for dim in 1..=2 {
            let length = self.mrope_section[dim] * 3;
            freqs_thw = freqs_thw.slice_assign(
                s![.., .., dim..length;3],
                freq.clone()
                    .slice(s![dim..dim+1, .., .., dim..length;3])
                    .squeeze_dim::<3>(0),
            );
        }
        Tensor::cat(vec![freqs_thw.clone(), freqs_thw], 2)
    }
}

#[derive(Config, Debug)]
pub struct TextRotaryEmbeddingConfig {}

impl TextRotaryEmbeddingConfig {
    pub fn init(&self, config: &TextConfig, device: &Device) -> TextRotaryEmbedding {
        let inv_freq = self.compute_default_rope_parameters(config, device);
        TextRotaryEmbedding {
            inv_freq,
            mrope_section: config.rope_parameters.mrope_section,
        }
    }

    fn compute_default_rope_parameters(&self, config: &TextConfig, device: &Device) -> Tensor<1> {
        let base = config.rope_parameters.rope_theta as f32;
        let partial_rotary_factor = config.rope_parameters.partial_rotary_factor;
        let head_dim = config.head_dim;
        let dim = head_dim as f32 * partial_rotary_factor;
        // 1/(base^(something)) = base^(-something) = e^ln(-something)
        let scaling_factor = -base.ln() / dim;
        (Tensor::arange_step(0..dim as i64, 2, device).float() * scaling_factor).exp()
    }
}

// activation: silu
#[derive(Module, Debug)]
pub struct RMSNormGated {
    weight: Param<Tensor<1>>, // (hidden_size,)
    variance_epsilon: f32,
}

impl RMSNormGated {
    pub fn forward(&self, hidden_states: Tensor<3>, gate: Tensor<3>) -> Tensor<3> {
        // hidden_states: (B, T, hidden_size)
        // gate: (B, T, hidden_size)
        let variance = hidden_states.clone().square().mean_dim(2); // (B, T, 1)
        let hidden_states = hidden_states * (variance + self.variance_epsilon).sqrt().recip(); // (B, T, hidden_size)
        let weight = self.weight.val().unsqueeze_dims::<3>(&[0, 1]); // (1, 1, hidden_size)
        let hidden_states = weight * hidden_states; // (B, T, hidden_size)
        hidden_states * silu(gate) // (B, T, hidden_size)
    }
}

#[derive(Config, Debug)]
pub struct RMSNormGatedConfig {
    hidden_size: usize,
    eps: f32,
}

impl RMSNormGatedConfig {
    pub fn init(&self, device: &Device) -> RMSNormGated {
        let weight = Param::from_tensor(Tensor::<1>::ones([self.hidden_size], device));
        RMSNormGated {
            weight,
            variance_epsilon: self.eps,
        }
    }
}

/// Keeps hidden states for which the mask is true
fn apply_mask_to_padding_states(
    hidden_states: Tensor<3>,
    attention_mask: Option<Tensor<2, Bool>>,
) -> Tensor<3> {
    // hidden_states: (B, T, hidden_size)
    // attention_mask: (B, T)
    if let Some(mask) = attention_mask {
        let mask = mask.unsqueeze_dim::<3>(2).float(); // (B, T, 1)
        hidden_states * mask
    } else {
        hidden_states
    }
}

fn causal_conv1d_update(
    hidden_states: Tensor<3>,
    conv_state: Tensor<3>,
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
) -> (Tensor<3>, Tensor<3>) {
    // hidden_states: (B, channels, seq_len)
    // conv_state: (B, channels, state_len)
    // weight: (channels, kernel_size)
    // bias: (channels,)

    let [_, channels, seq_len] = hidden_states.dims();
    let state_len = conv_state.dims()[2];

    let hidden_states_new = Tensor::cat(vec![conv_state, hidden_states], 2); // (B, channels, seq_len + state_len)
    let conv_state = hidden_states_new
        .clone()
        .slice(s![.., .., -(state_len as i64)..]); // (B, channels, state_len)

    let out = conv1d(
        hidden_states_new,
        weight.clone().unsqueeze_dim::<3>(1),
        bias,
        ConvOptions::new([1], [0], [1], channels),
    );

    let out = out.slice(s![.., .., -(seq_len as i64)..]); // (B, channels, seq_len)
    (silu(out), conv_state)
}

fn causal_conv1d_fn(
    hidden_states: Tensor<3>,
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
) -> Tensor<3> {
    let [_, channels, seq_len] = hidden_states.dims();
    let padding = weight.dims()[1] - 1;

    let out = conv1d(
        hidden_states,
        weight.unsqueeze_dim::<3>(1),
        bias,
        ConvOptions::new([1], [padding], [1], channels),
    );
    
    let out = out.slice(s![.., .., 0..seq_len]);

    silu(out)
}
