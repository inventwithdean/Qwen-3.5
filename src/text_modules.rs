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

fn l2norm<const D: usize>(x: Tensor<D>, eps: f32) -> Tensor<D> {
    let inv_norm = (x.clone().square().sum_dim(D - 1) + eps).sqrt().recip();
    x * inv_norm
}

// use_qk_l2norm_in_kernel = true
fn burn_chunk_gated_delta_rule(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    decay: Tensor<3>,
    beta: Tensor<3>,
    chunk_size: usize,
    initial_state: Option<Tensor<4>>,
    output_final_state: bool,
) -> (Tensor<4>, Option<Tensor<4>>) {
    // query: (B, seq_len, num_k_heads, k_head_dim)
    // key: (B, seq_len, num_k_heads, k_head_dim)
    // value: (B, seq_len, num_v_heads, v_head_dim)
    // decay: (B, seq_len, num_v_heads)
    // beta: (B, seq_len, num_v_heads)
    // initial_state: (B, num_v_heads, k_head_dim, v_head_dim)

    let [batch_size, seq_len, num_k_heads, k_head_dim] = key.dims();
    let [_, _, num_v_heads, v_head_dim] = value.dims();
    let recurrent_state_shape = [batch_size, num_v_heads, k_head_dim, v_head_dim];

    let query = query.swap_dims(1, 2); // (B, num_k_heads, seq_len, k_head_dim)
    let key = key.swap_dims(1, 2); // (B, num_k_heads, seq_len, k_head_dim)
    let value = value.swap_dims(1, 2); // (B, num_v_heads, seq_len, v_head_dim)

    let beta = beta.swap_dims(1, 2); // (B, num_v_heads, seq_len)
    let decay = decay.swap_dims(1, 2); // (B, num_v_heads, seq_len)

    let query = l2norm(query, 1e-6);
    let key = l2norm(key, 1e-6);

    let scaling = (k_head_dim as f32).powf(-0.5);
    let query = query * scaling;

    let pad_size = (chunk_size - seq_len % chunk_size) % chunk_size;

    // Burn applies from 1st dim to last dim, and if we provide it less than Dim, it applies to last dims.
    // So it applies them to seq_len and k_head_dim respectively.
    let qkv_padding = [(0, pad_size), (0, 0)];
    let query = query.pad(qkv_padding, 0.0);
    let key = key.pad(qkv_padding, 0.0);
    let value = value.pad(qkv_padding, 0.0);

    // Burn applies them to seq_len dim
    let bd_padding = [(0, pad_size)];
    let beta = beta.pad(bd_padding, 0.0);
    let decay = decay.pad(bd_padding, 0.0);

    let total_seq_len = seq_len + pad_size;
    let num_chunks = total_seq_len / chunk_size;

    let k_beta = key.clone() * beta.clone().unsqueeze_dim::<4>(3);
    let v_beta = value * beta.unsqueeze_dim::<4>(3);

    let query = query.reshape([batch_size, num_k_heads, num_chunks, chunk_size, k_head_dim]);
    let key = key.reshape([batch_size, num_k_heads, num_chunks, chunk_size, k_head_dim]);
    let k_beta = k_beta.reshape([batch_size, num_k_heads, num_chunks, chunk_size, k_head_dim]);

    let v_beta = v_beta.reshape([batch_size, num_v_heads, num_chunks, chunk_size, v_head_dim]);

    let decay = decay.reshape([batch_size, num_v_heads, num_chunks, chunk_size]);

    let strictly_upper_mask =
        Tensor::<2, Bool>::triu_mask([chunk_size, chunk_size], 1, &query.device());

    let cum_decay = decay.cumsum(3); // (batch_size, num_v_heads, num_chunks, chunk_size)

    // (B, num_v_heads, num_chunks, chunk_size, 1) - (B, num_v_heads, num_chunks, 1, chunk_size)
    // => (B, num_v_heads, num_chunks, chunk_size, chunk_size)
    let pairwise_decay =
        cum_decay.clone().unsqueeze_dim::<5>(4) - cum_decay.clone().unsqueeze_dim::<5>(3);
    let pairwise_decay = pairwise_decay.mask_fill(
        strictly_upper_mask.unsqueeze_dims::<5>(&[0, 1, 2]), // (1, 1, 1, chunk_size, chunk_size)
        f32::NEG_INFINITY,
    );
    let pairwise_decay = pairwise_decay.exp();

    // (B, num_k_heads, num_chunks, chunk_size, k_head_dim) @ (B, num_k_heads, num_chunks, k_head_dim, chunk_size)
    // => (B, num_k_heads, num_chunks, chunk_size, chunk_size)
    let ut_system = (k_beta.clone().matmul(key.clone().swap_dims(3, 4))) * pairwise_decay.clone();
    // (B, num_k_heads, num_chunks, chunk_size, k_head_dim) @ (B, num_k_heads, num_chunks, k_head_dim, chunk_size)
    // => (B, num_k_heads, num_chunks, chunk_size, chunk_size)
    let intra_chunk_attn = (query.clone().matmul(key.clone().swap_dims(3, 4))) * pairwise_decay;

    // num_k_heads and num_v_heads are same
    // (B, num_k_heads, num_chunks, chunk_size, k_head_dim) * (B, num_v_heads, num_chunks, chunk_size, 1)
    // => (B, num_k_heads, num_chunks, chunk_size, k_head_dim)
    let decayed_k_beta = k_beta * cum_decay.clone().exp().unsqueeze_dim::<5>(4);

    let mut ut_system = -ut_system.tril(-1); // (B, num_k_heads, num_chunks, chunk_size, chunk_size)
    for i in 1..chunk_size {
        let row = ut_system.clone().slice(s![.., .., .., i..i + 1, 0..i]); // (B, num_k_heads, num_chunks, 1, i)

        let sub = ut_system.clone().slice(s![.., .., .., 0..i, 0..i]); // (B, num_k_heads, num_chunks, i, i)
        let to_assign = row.clone() + row.matmul(sub); // (B, num_k_heads, num_chunks, 1, i)

        ut_system = ut_system.slice_assign(s![.., .., .., i..i + 1, ..i], to_assign);
    }

    ut_system =
        ut_system.clone() + Tensor::eye(chunk_size, &ut_system.device()).unsqueeze_dims(&[0, 1, 2]);

    let new_values = ut_system.clone().matmul(v_beta); // (B, num_k_heads, num_chunks, chunk_size, v_head_dim)
    let k_cumdecay = ut_system.clone().matmul(decayed_k_beta); // (B, num_k_heads, num_chunks, chunk_size, k_head_dim)

    let mut last_recurrent_state = match initial_state {
        Some(state) => state,
        None => Tensor::zeros(recurrent_state_shape, &ut_system.device()),
    };

    let mut core_attn_out = Tensor::zeros_like(&new_values);

    let query = query * cum_decay.clone().exp().unsqueeze_dim::<5>(4);
    let cum_decay_last = cum_decay
        .clone()
        .slice(s![.., .., .., (chunk_size - 1)..chunk_size]);

    let key = key
        * (cum_decay_last.clone() - cum_decay)
            .exp()
            .unsqueeze_dim::<5>(4);

    let chunk_decay = cum_decay_last.exp().unsqueeze_dim::<5>(4);

    for i in 0..num_chunks {
        let v_new = new_values
            .clone()
            .slice(s![.., .., i..i + 1, .., ..])
            .squeeze_dim::<4>(2)
            - k_cumdecay
                .clone()
                .slice(s![.., .., i..i + 1, .., ..])
                .squeeze_dim::<4>(2)
                .matmul(last_recurrent_state.clone());
        let inter_chunk_attn = query
            .clone()
            .slice(s![.., .., i..i + 1, .., ..])
            .squeeze_dim::<4>(2)
            .matmul(last_recurrent_state.clone());
        let to_assign = inter_chunk_attn
            + intra_chunk_attn
                .clone()
                .slice(s![.., .., i..i + 1, .., ..])
                .squeeze_dim::<4>(2)
                .matmul(v_new.clone());
        core_attn_out = core_attn_out.slice_assign(
            s![.., .., i..i + 1, .., ..],
            to_assign.unsqueeze_dim::<5>(2),
        );
        last_recurrent_state = last_recurrent_state
            * chunk_decay
                .clone()
                .slice(s![.., .., i..i + 1, .., ..])
                .squeeze_dim::<4>(2)
            + key
                .clone()
                .slice(s![.., .., i..i + 1, .., ..])
                .squeeze_dim::<4>(2)
                .swap_dims(2, 3)
                .matmul(v_new);
    }

    let last_recurrent_state = if output_final_state {
        Some(last_recurrent_state)
    } else {
        None
    };

    let core_attn_out = core_attn_out.reshape([batch_size, num_v_heads, total_seq_len, v_head_dim]);
    let core_attn_out = core_attn_out.slice(s![.., .., 0..seq_len, ..]);
    let core_attn_out = core_attn_out.swap_dims(1, 2);

    (core_attn_out, last_recurrent_state)
}
