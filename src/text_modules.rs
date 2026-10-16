use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::{Device, Int, s},
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
