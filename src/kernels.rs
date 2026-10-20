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
/// We will launch n * D kernels
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

#[backend_extension(Cube, Fusion)]
pub trait QuantEmbedding: Backend {
    #[fusion(meta=|table: &TensorSpec, ids: &TensorSpec| {
        let n = ids.shape.num_elements();
        let [_v, dim] = table.shape.dims::<2>();
        TensorSpec::new(Shape::from([n, dim]), DType::F32)
    })]
    fn quant_embedding(table: QuantizedTensor<Self>, ids: IntTensor<Self>) -> FloatTensor<Self>;
}

impl QuantEmbedding for CubeBackend {
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
    let out = <Dispatch as QuantEmbedding>::quant_embedding(
        embeddings.into_dispatch(),
        input_ids.into_dispatch(),
    );
    let out = burn::Tensor::<2>::from_dispatch(out);
    out
}
