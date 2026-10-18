use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{
        Linear, LinearConfig, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{
        Bool, Device, Distribution, Int,
        activation::{sigmoid, silu},
        module::{attention, conv1d},
        ops::{AttentionModuleOptions, ConvOptions},
        s,
    },
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

fn burn_recurrent_gated_delta_rule(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    decay: Tensor<3>,
    beta: Tensor<3>,
    initial_state: Option<Tensor<4>>,
    output_final_state: bool,
) -> (Tensor<4>, Option<Tensor<4>>) {
    // query: (B, seq_len, num_k_heads, k_head_dim)
    // key: (B, seq_len, num_k_heads, k_head_dim)
    // value: (B, seq_len, num_v_heads, v_head_dim)
    // decay: (B, seq_len, num_v_heads)
    // beta: (B, seq_len, num_v_heads)
    // initial_state: (B, num_v_heads, k_head_dim, v_head_dim)

    let [batch_size, seq_len, _, k_head_dim] = key.dims();
    let [_, _, num_v_heads, v_head_dim] = value.dims();

    let query = query.swap_dims(1, 2); // (B, num_k_heads, seq_len, k_head_dim)
    let key = key.swap_dims(1, 2); // (B, num_k_heads, seq_len, k_head_dim)
    let value = value.swap_dims(1, 2); // (B, num_v_heads, seq_len, v_head_dim)
    let decay = decay.swap_dims(1, 2); // (B, num_v_heads, seq_len)
    let beta = beta.swap_dims(1, 2); // (B, num_v_heads, seq_len)

    let query = l2norm(query, 1e-6);
    let key = l2norm(key, 1e-6);

    let query = query * (k_head_dim as f32).powf(-0.5);

    let mut last_recurrent_state = match initial_state {
        Some(state) => state,
        None => {
            let recurrent_state_shape = [batch_size, num_v_heads, k_head_dim, v_head_dim];
            Tensor::zeros(recurrent_state_shape, &query.device())
        }
    };

    let mut core_attn_out = Tensor::zeros_like(&value); // (B, num_v_heads, seq_len, v_head_dim)

    for i in 0..seq_len {
        let q_t = query
            .clone()
            .slice(s![.., .., i..i + 1, ..])
            .squeeze_dim::<3>(2); // (B, num_k_heads, k_head_dim)
        let k_t = key.clone().slice(s![.., .., i..i + 1]).squeeze_dim::<3>(2); // (B, num_k_heads, k_head_dim)
        let v_t = value
            .clone()
            .slice(s![.., .., i..i + 1])
            .squeeze_dim::<3>(2); // (B, num_v_heads, v_head_dim)

        let decay_t = decay
            .clone()
            .slice(s![.., .., i..i + 1])
            .exp()
            .unsqueeze_dim::<4>(3); // (B, num_v_heads, 1, 1)

        last_recurrent_state = last_recurrent_state * decay_t;
        let beta_t = beta.clone().slice(s![.., .., i..i + 1]); // (B, num_v_heads, 1)
        let kv_mem = (last_recurrent_state.clone() * k_t.clone().unsqueeze_dim::<4>(3))
            .sum_dim(2) // (B, num_v_heads, 1, v_head_dim)
            .squeeze_dim::<3>(2); // (B, num_v_heads, v_head_dim)
        let delta = (v_t - kv_mem) * beta_t; // (B, num_v_heads, v_head_dim)
        last_recurrent_state =
            last_recurrent_state + k_t.unsqueeze_dim::<4>(3) * delta.unsqueeze_dim::<4>(2);

        let to_assign = (last_recurrent_state.clone() * q_t.unsqueeze_dim::<4>(3)).sum_dim(2);

        core_attn_out = core_attn_out.slice_assign(s![.., .., i..i + 1, ..], to_assign);
    }

    let last_recurrent_state = if output_final_state {
        Some(last_recurrent_state)
    } else {
        None
    };

    core_attn_out = core_attn_out.swap_dims(1, 2);
    (core_attn_out, last_recurrent_state)
}

pub struct Cache {
    key_states: Vec<Option<Tensor<4>>>,
    value_states: Vec<Option<Tensor<4>>>,
    conv_states: Vec<Option<Tensor<3>>>,
    recurrent_states: Vec<Option<Tensor<4>>>,
}

impl Cache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            key_states: vec![None; num_layers],
            value_states: vec![None; num_layers],
            conv_states: vec![None; num_layers],
            recurrent_states: vec![None; num_layers],
        }
    }
}

#[derive(Module, Debug)]
pub struct GatedDeltaNet {
    conv1d: Conv1d,
    dt_bias: Param<Tensor<1>>,
    // Named as A_log, see if safetensors mapping fails.
    a_log: Param<Tensor<1>>,
    norm: RMSNormGated,
    out_proj: Linear,
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    layer_idx: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
}

impl GatedDeltaNet {
    fn forward(
        &self,
        hidden_states: Tensor<3>,
        attention_mask: Option<Tensor<2, Bool>>,
        mut cache: Option<&mut Cache>,
    ) -> Tensor<3> {
        let hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask);

        let [batch_size, seq_len, _] = hidden_states.dims();

        let mixed_qkv = self.in_proj_qkv.forward(hidden_states.clone());
        let mut mixed_qkv = mixed_qkv.swap_dims(1, 2);

        let z = self.in_proj_z.forward(hidden_states.clone());
        let z = z.reshape([batch_size, seq_len, self.num_v_heads, self.head_v_dim]);

        let b = self.in_proj_b.forward(hidden_states.clone());
        let a = self.in_proj_a.forward(hidden_states.clone());

        let use_precomputed_states = cache
            .as_ref()
            .map_or(false, |c| c.conv_states[self.layer_idx].is_some());

        if use_precomputed_states && seq_len == 1 {
            let conv_state = cache.as_ref().unwrap().conv_states[self.layer_idx]
                .clone()
                .unwrap();

            let (out, new_conv_state) = causal_conv1d_update(
                hidden_states,
                conv_state,
                self.conv1d.weight.clone().val().squeeze_dim(1),
                self.conv1d.bias.clone().map(|bias| bias.val()),
            );
            mixed_qkv = out;
            if let Some(c) = cache.as_mut() {
                c.conv_states[self.layer_idx] = Some(new_conv_state);
            }
        } else {
            let mut input_mixed_qkv = mixed_qkv.clone();

            if let Some(c) = cache.as_mut() {
                if let Some(prev_state) = &c.conv_states[self.layer_idx] {
                    input_mixed_qkv = Tensor::cat(vec![prev_state.clone(), input_mixed_qkv], 2);
                }
                let padding = (self.conv1d.weight.dims()[2] - 1) as i64;
                c.conv_states[self.layer_idx] =
                    Some(input_mixed_qkv.clone().slice(s![.., .., -padding..]));
            }

            mixed_qkv = causal_conv1d_fn(
                input_mixed_qkv,
                self.conv1d.weight.val().squeeze_dim(1),
                self.conv1d.bias.clone().map(|bias| bias.val()),
            );

            if cache.is_some() {
                mixed_qkv = mixed_qkv.slice(s![.., .., -(seq_len as i64)..]);
            }
        }

        mixed_qkv = mixed_qkv.swap_dims(1, 2); // (B, seq_len, key_dim * 2 + value_dim)

        let key_dim = self.num_k_heads * self.head_k_dim;
        let value_dim = self.num_v_heads * self.head_v_dim;

        let query = mixed_qkv.clone().slice(s![.., .., 0..key_dim]);
        let key = mixed_qkv.clone().slice(s![.., .., key_dim..(key_dim * 2)]);
        let value = mixed_qkv
            .clone()
            .slice(s![.., .., (key_dim * 2)..(key_dim * 2 + value_dim)]);

        let mut query = query.reshape([batch_size, seq_len, self.num_k_heads, self.head_k_dim]);
        let mut key = key.reshape([batch_size, seq_len, self.num_k_heads, self.head_k_dim]);
        let value = value.reshape([batch_size, seq_len, self.num_v_heads, self.head_v_dim]);

        let beta = sigmoid(b);

        let a_log_exp = self.a_log.val().exp().unsqueeze_dims::<3>(&[0, 1]); // (1, 1, num_v_heads)
        let dt_bias = self.dt_bias.val().unsqueeze_dims::<3>(&[0, 1]); // (1, 1, num_v_heads)
        let softplus = (a + dt_bias).exp().add_scalar(1.0).log();
        let g = -a_log_exp * softplus;

        if self.num_v_heads / self.num_k_heads > 1 {
            let repeat_factor = self.num_v_heads / self.num_k_heads;

            let expanded_shape = [
                batch_size,
                seq_len,
                self.num_k_heads,
                repeat_factor,
                self.head_k_dim,
            ];
            let interleaved_shape = [batch_size, seq_len, self.num_v_heads, self.head_k_dim];
            query = query
                .unsqueeze_dim::<5>(3)
                .expand(expanded_shape)
                .reshape(interleaved_shape);
            key = key
                .unsqueeze_dim::<5>(3)
                .expand(expanded_shape)
                .reshape(interleaved_shape);
        }

        let recurrent_state = if use_precomputed_states {
            cache.as_ref().unwrap().recurrent_states[self.layer_idx].clone()
        } else {
            None
        };

        let output_final_state = cache.is_some();

        let (core_attn_out, last_recurrent_state) = if use_precomputed_states && seq_len == 1 {
            burn_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                recurrent_state,
                output_final_state,
            )
        } else {
            burn_chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                64,
                recurrent_state,
                output_final_state,
            )
        };

        if let Some(c) = cache {
            c.recurrent_states[self.layer_idx] = last_recurrent_state;
        }
        let num_elements = batch_size * seq_len * self.num_v_heads;
        let core_attn_out = core_attn_out.reshape([num_elements, self.head_v_dim]);
        let z = z.reshape([num_elements, self.head_v_dim]);
        let core_attn_out = self
            .norm
            .forward(core_attn_out.unsqueeze_dim::<3>(1), z.unsqueeze_dim::<3>(1));
        let core_attn_out = core_attn_out.reshape([batch_size, seq_len, value_dim]);
        self.out_proj.forward(core_attn_out)
    }
}

#[derive(Config, Debug)]
pub struct GatedDeltaNetConfig {
    layer_idx: usize,
}

impl GatedDeltaNetConfig {
    pub fn init(&self, config: &TextConfig, device: &Device) -> GatedDeltaNet {
        let hidden_size = config.hidden_size;
        let num_v_heads = config.linear_num_value_heads;
        let num_k_heads = config.linear_num_key_heads;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;

        let conv_kernel_size = config.linear_conv_kernel_dim;
        let layer_idx = self.layer_idx;
        let layer_norm_epsilon = config.rms_norm_eps;
        let conv_dim = key_dim * 2 + value_dim;

        let conv1d = Conv1dConfig::new(conv_dim, conv_dim, conv_kernel_size)
            .with_bias(false)
            .with_groups(conv_dim)
            .with_padding(PaddingConfig1d::Explicit(
                conv_kernel_size - 1,
                conv_kernel_size - 1,
            ))
            .init(device);
        let dt_bias = Param::from_tensor(Tensor::<1>::ones([num_v_heads], device));
        let a = Tensor::<1>::random([num_v_heads], Distribution::Uniform(0.01, 16.0), device);
        let a_log = Param::from_tensor(a.log());

        let norm = RMSNormGatedConfig::new(head_v_dim, layer_norm_epsilon).init(device);
        let out_proj = LinearConfig::new(value_dim, hidden_size)
            .with_bias(false)
            .init(device);

        let in_proj_qkv = LinearConfig::new(hidden_size, key_dim * 2 + value_dim)
            .with_bias(false)
            .init(device);
        let in_proj_z = LinearConfig::new(hidden_size, value_dim)
            .with_bias(false)
            .init(device);
        let in_proj_b = LinearConfig::new(hidden_size, num_v_heads)
            .with_bias(false)
            .init(device);
        let in_proj_a = LinearConfig::new(hidden_size, num_v_heads)
            .with_bias(false)
            .init(device);

        GatedDeltaNet {
            conv1d,
            dt_bias,
            a_log,
            norm,
            out_proj,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            layer_idx,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        }
    }
}

fn rotate_half(x: Tensor<4>) -> Tensor<4> {
    // x: (B, num_heads, seq_len, head_dim)
    let head_dim = x.dims()[3];

    let x1 = x.clone().slice(s![.., .., .., 0..head_dim / 2]);
    let x2 = x.slice(s![.., .., .., head_dim / 2..]);

    Tensor::cat(vec![-x2, x1], 3)
}

fn apply_rotary_pos_emb(
    q: Tensor<4>,
    k: Tensor<4>,
    cos: Tensor<3>,
    sin: Tensor<3>,
) -> (Tensor<4>, Tensor<4>) {
    // q: (batch_size, num_heads, seq_len, head_dim)
    // k: (batch_size, num_kv_heads, seq_len, head_dim)
    // cos, sin: (batch_size, seq_len, rotary_dim)

    let cos = cos.unsqueeze_dim::<4>(1); // (batch_size, 1, seq_len, rotary_dim)
    let sin = sin.unsqueeze_dim::<4>(1); // (batch_size, 1, seq_len, rotary_dim)

    let rotary_dim = cos.dims()[3];

    let q_rot = q.clone().slice(s![.., .., .., 0..rotary_dim]);
    let q_pass = q.slice(s![.., .., .., rotary_dim..]);
    let k_rot = k.clone().slice(s![.., .., .., 0..rotary_dim]);
    let k_pass = k.slice(s![.., .., .., rotary_dim..]);

    let q_embed = (q_rot.clone() * cos.clone()) + (rotate_half(q_rot) * sin.clone());
    let k_embed = (k_rot.clone() * cos) + (rotate_half(k_rot) * sin);
    let q_embed = Tensor::cat(vec![q_embed, q_pass], 3);
    let k_embed = Tensor::cat(vec![k_embed, k_pass], 3);
    (q_embed, k_embed)
}

fn repeat_kv(hidden_states: Tensor<4>, repeats: usize) -> Tensor<4> {
    // hidden_states: (batch_size, num_key_value_heads, seq_len, head_dim)
    let [b, num_key_value_heads, seq_len, head_dim] = hidden_states.dims();
    if repeats == 1 {
        hidden_states
    } else {
        // (batch_size, num_key_value_heads, 1, seq_len, head_dim)
        let hidden_states = hidden_states.unsqueeze_dim::<5>(2).expand([
            b,
            num_key_value_heads,
            repeats,
            seq_len,
            head_dim,
        ]);
        hidden_states.reshape([b, num_key_value_heads * repeats, seq_len, head_dim])
    }
}

#[derive(Module, Debug)]
pub struct RMSNorm {
    weight: Param<Tensor<1>>,
    eps: f32,
}

impl RMSNorm {
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let output = x.clone() * (x.square().mean_dim(D - 1) + self.eps).sqrt().recip();
        let mut weight_shape = [1; D];
        weight_shape[D - 1] = self.weight.dims()[0];

        let output = output * (1.0 + self.weight.val().reshape(weight_shape));
        output
    }
}

#[derive(Config, Debug)]
pub struct RMSNormConfig {
    dim: usize,
    eps: f32,
}

impl RMSNormConfig {
    pub fn init(&self, device: &Device) -> RMSNorm {
        RMSNorm {
            weight: Param::from_tensor(Tensor::zeros([self.dim], device)),
            eps: self.eps,
        }
    }
}

#[derive(Module, Debug)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,

    head_dim: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
}

impl Attention {
    pub fn forward(
        &self,
        hidden_states: Tensor<3>,
        position_embeddings: (Tensor<3>, Tensor<3>),
        attention_mask: Option<Tensor<4, Bool>>,
        mut past_key_values: Option<&mut Cache>,
    ) -> Tensor<3> {
        // hidden_states: (batch_size, seq_len, head_dim)
        // both position_embeddings: (batch_size, seq_len, head_dim)
        // attention_mask: (batch_size, seq_len)
        let [batch_size, seq_len, _] = hidden_states.dims();

        let q_proj = self.q_proj.forward(hidden_states.clone()).reshape([
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.head_dim * 2,
        ]);
        let query_states_gate = q_proj.chunk(2, 3);
        let query_states = query_states_gate[0].clone(); // (batch_size, seq_len, num_attention_heads, head_dim)
        let gate = query_states_gate[1].clone(); // (batch_size, seq_len, num_attention_heads, head_dim)

        let gate = gate.reshape([
            batch_size,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ]);
        let query_states = self.q_norm.forward(query_states).swap_dims(1, 2);
        let key_states = self.k_proj.forward(hidden_states.clone()).reshape([
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ]);
        let key_states = self.k_norm.forward(key_states).swap_dims(1, 2);

        let value_states = self.v_proj.forward(hidden_states).reshape([
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ]);
        let value_states = value_states.swap_dims(1, 2);

        let (cos, sin) = position_embeddings;
        let (query_states, key_states) = apply_rotary_pos_emb(query_states, key_states, cos, sin);

        let num_key_value_groups = self.num_attention_heads / self.num_key_value_heads;
        let key_states = repeat_kv(key_states, num_key_value_groups);
        let value_states = repeat_kv(value_states, num_key_value_groups);

        // As of now, query state is: (batch_size, num_attention_heads, seq_len_q, head_dim)
        // and key and value states: (batch_size, num_attention_heads, seq_len_k, head_dim)
        // attention expects (batch_size, num_heads, seq_len, head_dim)
        // In input attention mask, true means attend.
        let attention_mask = attention_mask.map(|mask| mask.bool_not());

        let attn_output = attention(
            query_states,
            key_states,
            value_states,
            attention_mask, // true means mask.
            None,
            AttentionModuleOptions::default(),
        ); // (batch_size, num_attention_heads, seq_len_q, head_dim)

        let attn_output = attn_output.swap_dims(1, 2); // (batch_size, seq_len, num_attention_heads, head_dim)
        let attn_output = attn_output.reshape([
            batch_size,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ]);
        let attn_output = attn_output * sigmoid(gate);
        self.o_proj.forward(attn_output)
    }
}

#[derive(Config, Debug)]
pub struct AttentionConfig {}

impl AttentionConfig {
    pub fn init(&self, config: &TextConfig, device: &Device) -> Attention {
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;
        let attn_bias = config.attention_bias;
        let rms_norm_eps = config.rms_norm_eps;

        let q_proj = LinearConfig::new(hidden_size, num_attention_heads * head_dim * 2)
            .with_bias(attn_bias)
            .init(device);
        let k_proj = LinearConfig::new(hidden_size, num_key_value_heads * head_dim)
            .with_bias(attn_bias)
            .init(device);
        let v_proj = LinearConfig::new(hidden_size, num_key_value_heads * head_dim)
            .with_bias(attn_bias)
            .init(device);
        let o_proj = LinearConfig::new(num_attention_heads * head_dim, hidden_size)
            .with_bias(attn_bias)
            .init(device);
        let q_norm = RMSNormConfig::new(head_dim, rms_norm_eps).init(device);
        let k_norm = RMSNormConfig::new(head_dim, rms_norm_eps).init(device);

        Attention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            head_dim,
            num_attention_heads,
            num_key_value_heads,
        }
    }
}
