use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata, backend_extension,
        fusion::custom::TensorSpec,
        ops::FloatTensorOps,
        tensor::{FloatTensor, IntTensor, QuantizedTensor},
        wgpu::CubeBackend,
    },
    tensor::{DType, FloatDType, Shape},
};
use cubecl::prelude::*;

/// This only works for Q8F quantized embeddings
/// We will launch n * D units
#[cube(launch)]
fn q8_embedding_lookup_kernel(
    table: &Tensor<u32>,      // [vocab_size, D / 4] packed Q8F words, flat
    ids: &Tensor<i32>,        // [n]
    scales: &Tensor<f32>,     // [vocab_size, D / 64]
    output: &mut Tensor<f32>, // [n, D]
    #[comptime] dim: usize,   // D
) {
    let pos = ABSOLUTE_POS;
    // guard check to stay less than n*D
    if pos < output.len() {
        let row = pos / dim; // Units 0 to D-1 cover first row, D to 2D-1 cover the second and so on
        let col = pos % dim; // For 1st row, unit 0 covers first embedding idx, unit 1 covers second and so on
        let id = ids[row]; // Which id this unit is working on, each id will have D units
        let words_per_row = dim / 4; // Q8F
        // e.g. id = 1, words_per_row = 512 (dim = 2048), col = 3
        // word_idx = 512 + 3 / 4 = 512, which is what we want. as each word contains 4 values, (col 0, 1, 2 and 3 all will go to 512th word)
        let word_idx = usize::cast_from(id) * words_per_row + col / 4;
        let word = table[word_idx];
        // col 0 -> 0 shift
        // col 1 -> 8 shift
        // col 2 -> 16 shift
        // col 3 -> 24 shift
        // Because [a, b, c, d] are stored as [d, c, b, a] in packed format.
        let shift = u32::cast_from(col % 4) * 8u32;
        let raw = (word >> shift) & 255u32; // (Masking with 255 to get actual value)
        let is_negative = raw >= 128u32; // We have to subtract 256 if we originally had a negative number
        let signed = i32::cast_from(raw) - i32::cast_from(is_negative) * 256;
        output[pos] = f32::cast_from(signed) * scales[usize::cast_from(id) * (dim / 64) + col / 64];
    }
}

/// e.g. in_proj_qkv: [2048, 6144]
/// weights: [2048, 1536] Q8F packed words
/// hidden_states: [2048]
/// scales: [2048, 96]
/// output: [6144]
/// We will launch out_dim units, one for each output
#[cube(launch)]
fn q8_gemv_kernel(
    hidden_states: &Tensor<f32>, // [in_dim]
    weights: &Tensor<u32>,       // [in_dim, out_dim / 4]
    scales: &Tensor<f32>,        // [in_dim, out_dim / 64]
    output: &mut Tensor<f32>,    // [out_dim]
    #[comptime] in_dim: usize,
    #[comptime] out_dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!()
    }
    // pos 0: shift 0
    // pos 1: shift 8
    // pos 2: shift 16
    // pos 3: shift 24
    // pos 4: shift 0 ...
    let shift = u32::cast_from(pos % 4) * 8;
    let col = pos / 4; // 0 to 3 work on 1st col of weights, 4-7 work on 2nd col of weights and so on...
    let words_per_row = out_dim / 4;
    let mut acc = 0.0f32;
    let num_scales = out_dim / 64;
    let scale_col = pos / 64;
    let u = 8usize;
    for blk in 0..(in_dim / u) {
        #[unroll]
        for j in 0..u {
            let row = blk * u + j;
            let word_idx = row * words_per_row + col;
            let word = weights[word_idx];
            let raw = (word >> shift) & 255u32; // Get correct bytes
            let is_negative = raw >= 128u32;
            let signed = i32::cast_from(raw) - i32::cast_from(is_negative) * 256;
            let scale_idx = row * num_scales + scale_col;
            acc += f32::cast_from(signed) * hidden_states[row] * scales[scale_idx];
        }
    }
    output[pos] = acc;
}

/// Will launch out_dim * num_splits units
/// CubeCount: (num_tiles, num_splits, 1)
/// e.g. in_proj_qkv: [2048, 6144]
/// weights: [2048, 1536] Q8F packed words
/// hidden_states: [2048]
/// scales: [2048, 96]
/// num_splits = 4
/// output: [4, 6144]
/// Every thread will now read the chunk which is in_dim / 4 => 512
#[cube(launch)]
fn q8_gemv_partial_sum_kernel(
    hidden_states: &Tensor<f32>, // [in_dim]
    weights: &Tensor<u32>,       // [in_dim, out_dim / 4]
    scales: &Tensor<f32>,        // [in_dim, out_dim / 64]
    output: &mut Tensor<f32>,    // [num_splits, out_dim]
    #[comptime] in_dim: usize,
    #[comptime] out_dim: usize,
    #[comptime] num_splits: usize,
) {
    let pos = usize::cast_from(CUBE_POS_X * 256 + UNIT_POS); // 0..6143, the output idx
    if pos >= out_dim {
        terminate!()
    }
    let split = usize::cast_from(CUBE_POS_Y);

    let chunk_size = in_dim / num_splits; // 512
    let row_start = chunk_size * split;

    let words_per_row = out_dim / 4;
    let mut acc = 0.0f32;
    let num_scales = out_dim / 64;

    let shift = (u32::cast_from(pos) % 4) * 8u32;
    let col = pos / 4; // Q8F, 6144 units should cover 1536 words
    let scale_col = pos / 64;
    let u = 8usize;
    for blk in 0..(chunk_size / u) {
        #[unroll]
        for j in 0..u {
            let i = row_start + blk * u + j;
            let word_idx = words_per_row * i + col;
            let word = weights[word_idx];
            let raw = (word >> shift) & 255u32;
            let is_negative = raw >= 128u32;
            let signed = i32::cast_from(raw) - i32::cast_from(is_negative) * 256;
            let scale_idx = i * num_scales + scale_col;
            acc += f32::cast_from(signed) * hidden_states[i] * scales[scale_idx];
        }
    }

    output[out_dim * split + pos] = acc;
}

/// Will launch dim units
#[cube(launch)]
fn q8_gemv_sum_reduction_kernel(
    input: &Tensor<f32>,      // [num_splits, dim]
    output: &mut Tensor<f32>, // [dim]
    #[comptime] dim: usize,
    #[comptime] num_splits: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= dim {
        terminate!()
    }
    let mut acc = 0.0f32;
    #[unroll]
    for i in 0..num_splits {
        acc += input[dim * i + pos];
    }
    output[pos] = acc;
}

// Will launch 2 * num_splits * n_heads units
// And 1 block
// e.g. hidden_states: [2048]
// w_b: [2048, 16]
// w_a: [2048, 16]
// a_log: [16]
// dt_bias: [16]
#[cube(launch)]
fn gate_proj_kernel(
    hidden_states: &Tensor<f32>, // [in_dim]
    w_b: &Tensor<f32>,           // [in_dim, n_heads]
    w_a: &Tensor<f32>,           // [in_dim, n_heads]
    a_log: &Tensor<f32>,         // [n_heads]
    dt_bias: &Tensor<f32>,       // [n_heads]
    out: &mut Tensor<f32>,       // [2 * n_heads] -> half beta, half gate
    #[comptime] in_dim: usize,
    #[comptime] n_heads: usize,
    #[comptime] num_splits: usize,
) {
    // beta then gate
    let mut partial = Shared::<[f32]>::new_slice(num_splits * 2 * n_heads); // (num_splits, 2 * n_heads)
    let n_out = 2 * n_heads;
    let pos = ABSOLUTE_POS;
    let output_idx = pos % n_heads; // 0 to 15
    let chunk_size = in_dim / num_splits;
    let split_idx = pos / n_out;
    let row_start = split_idx * chunk_size;
    let is_beta = (pos / n_heads) % 2 == 0; // Units 0 to 15 for beta, 16 to 31 for gate, 32 to 47 for beta, 48 to 61 for gate and so on
    let mut acc = 0.0f32;
    let u = 8usize;
    if is_beta {
        for blk in 0..(chunk_size / u) {
            #[unroll]
            for j in 0..u {
                let row = row_start + blk * u + j;
                let weight_idx = n_heads * row + output_idx;
                acc += hidden_states[row] * w_b[weight_idx];
            }
        }
        partial[2 * n_heads * split_idx + output_idx] = acc;
    } else {
        for blk in 0..(chunk_size / u) {
            #[unroll]
            for j in 0..u {
                let row = row_start + blk * u + j;
                let weight_idx = n_heads * row + output_idx;
                acc += hidden_states[row] * w_a[weight_idx];
            }
        }
        partial[2 * n_heads * split_idx + output_idx + n_heads] = acc;
    }
    sync_cube(); // barrier so that every partial sum is there
    if split_idx == 0 {
        if is_beta {
            // sigmoid(b) => 1 / (1 + exp(-b))
            let mut sum = 0.0f32; // b
            #[unroll]
            for i in 0..num_splits {
                let partial_idx = n_out * i + output_idx;
                sum += partial[partial_idx];
            }
            let sigmoid = 1.0 / (1.0 + (-sum).exp());
            out[output_idx] = sigmoid;
        } else {
            let mut sum = 0.0f32; // a
            #[unroll]
            for i in 0..num_splits {
                let partial_idx = n_out * i + output_idx + n_heads;
                sum += partial[partial_idx];
            }
            let x_gate = sum + dt_bias[output_idx];
            let pre_softplus = (x_gate.exp() + 1.0).ln();
            let softplus = if pre_softplus > 20.0 {
                x_gate
            } else {
                pre_softplus
            };
            out[output_idx + n_heads] = -a_log[output_idx].exp() * softplus;
        }
    }
}

#[backend_extension(Cube, Fusion)]
pub trait CustomKernels: Backend {
    #[fusion(meta=|table: &TensorSpec, ids: &TensorSpec| {
        let n = ids.shape.num_elements();
        let [_v, dim] = table.shape.dims::<2>();
        TensorSpec::new(Shape::from([n, dim]), DType::F32)
    })]
    fn q8_embedding_lookup(table: QuantizedTensor<Self>, ids: IntTensor<Self>)
    -> FloatTensor<Self>;
    #[fusion(meta=|_hidden_states: &TensorSpec, weights: &TensorSpec| {
        let [_in_dim, out_dim] = weights.shape.dims::<2>();
        TensorSpec::new(Shape::from([out_dim]), DType::F32)
    })]
    fn q8_gemv(
        hidden_states: FloatTensor<Self>,
        weights: QuantizedTensor<Self>,
    ) -> FloatTensor<Self>;
    #[fusion(meta=|_hidden_states: &TensorSpec, weights: &TensorSpec| {
        let [_in_dim, out_dim] = weights.shape.dims::<2>();
        TensorSpec::new(Shape::from([out_dim]), DType::F32)
    })]
    fn q8_gemv_partial_sum(
        hidden_states: FloatTensor<Self>,
        weights: QuantizedTensor<Self>,
    ) -> FloatTensor<Self>;

    #[fusion(meta=|_hidden_states: &TensorSpec, w_b: &TensorSpec, _w_a: &TensorSpec, _a_log: &TensorSpec, _dt_bias: &TensorSpec| {
        let [_in_dim, n_heads] = w_b.shape.dims::<2>();
        TensorSpec::new(Shape::from([2 * n_heads]), DType::F32)
    })]
    fn gate_proj(
        hidden_states: FloatTensor<Self>,
        w_b: FloatTensor<Self>,
        w_a: FloatTensor<Self>,
        a_log: FloatTensor<Self>,
        dt_bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

impl CustomKernels for CubeBackend {
    fn q8_embedding_lookup(
        table: QuantizedTensor<Self>,
        ids: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let (values, scales) = table
            .quantized_handles()
            .expect("Embedding table must be QFloat");
        let client = values.client.clone();
        let device = values.device.clone();

        let [_vocab, dim] = table.shape().dims::<2>();
        let [_b, n] = ids.shape().dims::<2>();

        // Have to import burn::backend::ops:FloatTensorOps
        let output = Self::float_empty(Shape::from([n, dim]), &device, FloatDType::F32);
        let num_elements = (n * dim) as u32;
        let cube_dim = CubeDim::new_3d(256, 1, 1);
        let cube_count = CubeCount::Static(num_elements.div_ceil(cube_dim.x), 1, 1);

        q8_embedding_lookup_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            values.into_tensor_arg(),
            ids.into_tensor_arg(),
            scales.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            dim,
        );

        output
    }
    fn q8_gemv(
        hidden_states: FloatTensor<Self>,
        weights: QuantizedTensor<Self>,
    ) -> FloatTensor<Self> {
        let (values, scales) = weights.quantized_handles().expect("weights must be QFloat");
        let client = values.client.clone();
        let device = values.device.clone();

        let [in_dim, out_dim] = weights.shape().dims::<2>();
        let output = Self::float_empty(Shape::from([out_dim]), &device, FloatDType::F32);
        let cube_dim = CubeDim::new_3d(256, 1, 1);
        let cube_count = CubeCount::Static((out_dim as u32).div_ceil(cube_dim.x), 1, 1);

        q8_gemv_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            hidden_states.into_tensor_arg(),
            values.into_tensor_arg(),
            scales.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            in_dim,
            out_dim,
        );
        output
    }

    fn q8_gemv_partial_sum(
        hidden_states: FloatTensor<Self>,
        weights: QuantizedTensor<Self>,
    ) -> FloatTensor<Self> {
        let (values, scales) = weights.quantized_handles().expect("weights must be QFloat");
        let client = values.client.clone();
        let device = values.device.clone();

        let [in_dim, out_dim] = weights.shape().dims::<2>();
        let num_splits = (25_000 / out_dim).clamp(1, (in_dim / 80).max(1));
        // Partial sum output
        let output =
            Self::float_empty(Shape::from([num_splits, out_dim]), &device, FloatDType::F32);
        let cube_dim = CubeDim::new_3d(256, 1, 1);
        let cube_count =
            CubeCount::Static((out_dim as u32).div_ceil(cube_dim.x), num_splits as u32, 1);

        q8_gemv_partial_sum_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            hidden_states.into_tensor_arg(),
            values.into_tensor_arg(),
            scales.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            in_dim,
            out_dim,
            num_splits,
        );
        let reduced = Self::float_empty(Shape::from([out_dim]), &device, FloatDType::F32);
        let reduce_count = CubeCount::Static((out_dim as u32).div_ceil(cube_dim.x), 1, 1);

        q8_gemv_sum_reduction_kernel::launch(
            &client,
            reduce_count,
            cube_dim,
            output.into_tensor_arg(),
            reduced.clone().into_tensor_arg(),
            out_dim,
            num_splits,
        );
        reduced
    }

    fn gate_proj(
        hidden_states: FloatTensor<Self>,
        w_b: FloatTensor<Self>,
        w_a: FloatTensor<Self>,
        a_log: FloatTensor<Self>,
        dt_bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let client = hidden_states.client.clone();
        let device = hidden_states.device.clone();
        let [in_dim, n_heads] = w_b.shape().dims::<2>();
        let output = Self::float_empty(Shape::from([2 * n_heads]), &device, FloatDType::F32);
        let num_splits = 8;

        let num_units = 2 * n_heads * num_splits;
        let cube_dim = CubeDim::new_3d(num_units as u32, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        gate_proj_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            hidden_states.into_tensor_arg(),
            w_b.into_tensor_arg(),
            w_a.into_tensor_arg(),
            a_log.into_tensor_arg(),
            dt_bias.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            in_dim,
            n_heads,
            num_splits,
        );

        output
    }
}

/// embeddings: (vocab_size, emb_dim)
///
/// input_ids: (batch_size, seq_len)
///
/// Currently only works with batch size 1.
/// Make sure embeddings are tensor-scale Q8F quantized
pub fn q8_embedding_lookup(
    embeddings: burn::Tensor<2>,
    input_ids: burn::Tensor<2, burn::tensor::Int>,
) -> burn::Tensor<2> {
    let out = <Dispatch as CustomKernels>::q8_embedding_lookup(
        embeddings.into_dispatch(),
        input_ids.into_dispatch(),
    );
    let out = burn::Tensor::<2>::from_dispatch(out);
    out
}

pub fn q8_gemv(hidden_states: burn::Tensor<3>, weights: burn::Tensor<2>) -> burn::Tensor<3> {
    let out = <Dispatch as CustomKernels>::q8_gemv(
        hidden_states.squeeze_dims::<1>(&[0, 1]).into_dispatch(),
        weights.into_dispatch(),
    );
    let out = burn::Tensor::<1>::from_dispatch(out).unsqueeze_dims::<3>(&[0, 1]);
    out
}

pub fn q8_gemv_partial_sum(
    hidden_states: burn::Tensor<3>,
    weights: burn::Tensor<2>,
) -> burn::Tensor<3> {
    let out = <Dispatch as CustomKernels>::q8_gemv_partial_sum(
        hidden_states.squeeze_dims::<1>(&[0, 1]).into_dispatch(),
        weights.into_dispatch(),
    );
    let out = burn::Tensor::<1>::from_dispatch(out).unsqueeze_dims::<3>(&[0, 1]);
    out
}

pub fn gate_proj(
    hidden_states: burn::Tensor<3>,
    w_b: burn::Tensor<2>,
    w_a: burn::Tensor<2>,
    a_log: burn::Tensor<1>,
    dt_bias: burn::Tensor<1>,
) -> (burn::Tensor<3>, burn::Tensor<3>) {
    let out = <Dispatch as CustomKernels>::gate_proj(
        hidden_states.squeeze_dims::<1>(&[0, 1]).into_dispatch(),
        w_b.into_dispatch(),
        w_a.into_dispatch(),
        a_log.into_dispatch(),
        dt_bias.into_dispatch(),
    );
    let out = burn::Tensor::<1>::from_dispatch(out);
    let beta_gate_vec = out.chunk(2, 0);
    (
        beta_gate_vec[0].clone().unsqueeze_dims::<3>(&[0, 1]),
        beta_gate_vec[1].clone().unsqueeze_dims::<3>(&[0, 1]),
    )
}
