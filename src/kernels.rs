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
fn embedding_lookup_kernel(
    table: &Tensor<u32>,      // [vocab_size, D / 4] packed Q8F words, flat
    ids: &Tensor<i32>,        // [n]
    scales: &Tensor<f32>,     // per-tensor scale
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
        output[pos] = f32::cast_from(signed) * scales[0];
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
    let row_end = row_start + chunk_size;

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
    if pos > dim {
        terminate!()
    }
    let mut acc = 0.0f32;
    #[unroll]
    for i in 0..num_splits {
        acc += input[dim * i + pos];
    }
    output[pos] = acc;
}

#[backend_extension(Cube, Fusion)]
pub trait CustomKernels: Backend {
    #[fusion(meta=|table: &TensorSpec, ids: &TensorSpec| {
        let n = ids.shape.num_elements();
        let [_v, dim] = table.shape.dims::<2>();
        TensorSpec::new(Shape::from([n, dim]), DType::F32)
    })]
    fn quant_embedding(table: QuantizedTensor<Self>, ids: IntTensor<Self>) -> FloatTensor<Self>;
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
}

impl CustomKernels for CubeBackend {
    fn quant_embedding(table: QuantizedTensor<Self>, ids: IntTensor<Self>) -> FloatTensor<Self> {
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

        embedding_lookup_kernel::launch(
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
}

/// embeddings: (vocab_size, emb_dim)
///
/// input_ids: (batch_size, seq_len)
///
/// Currently only works with batch size 1.
/// Make sure embeddings are tensor-scale Q8F quantized
pub fn quant_embeddings(
    embeddings: burn::Tensor<2>,
    input_ids: burn::Tensor<2, burn::tensor::Int>,
) -> burn::Tensor<2> {
    let out = <Dispatch as CustomKernels>::quant_embedding(
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
