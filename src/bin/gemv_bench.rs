use std::time::Instant;

use burn::tensor::{
    Device, Distribution, Tensor,
    quantization::{QuantScheme, ScaleDtype},
};
use qwen3_5::kernels::{q8_gemv, q8_gemv_partial_sum};

const IN_DIM: usize = 2048;
const OUT_DIM: usize = 6144;
const ITERS: usize = 300;
const WARMUP: usize = 30;
const N_WEIGHTS: usize = 4;

fn weight_bytes() -> f64 {
    let values = (IN_DIM * OUT_DIM / 4 * 4) as f64; // 4 x i8 per u32 word
    let scales = (IN_DIM * (OUT_DIM / 64) * 4) as f64; // one f32 per 64 values
    values + scales
}

fn main() {
    let device = Device::wgpu(Default::default());
    let bytes = weight_bytes();
    println!("per-weight footprint : {:.2} MB", bytes / 1e6);
    println!(
        "rotating working set : {:.1} MB   ({} weights)",
        bytes * N_WEIGHTS as f64 / 1e6,
        N_WEIGHTS
    );
    println!("iters                : {ITERS}   (+{WARMUP} warmup)\n");

    println!("allocating and quantizing {N_WEIGHTS} weights...");
    let scheme = QuantScheme::default().per_block([64], ScaleDtype::F32);
    let weights: Vec<Tensor<2>> = (0..N_WEIGHTS)
        .map(|_| {
            let w = Tensor::<2>::random([IN_DIM, OUT_DIM], Distribution::Default, &device)
                .quantize_dynamic(&scheme);
            device.sync().unwrap();
            w
        })
        .collect();
    println!("done\n");

    let x = Tensor::<3>::random([1, 1, IN_DIM], Distribution::Default, &device);

    measure("A  OLD kernel, one weight reused     (L2)", &device, |_| {
        let _ = q8_gemv(x.clone(), weights[0].clone());
    });

    measure("B  OLD kernel, rotating 4 weights   (DRAM)", &device, |i| {
        let _ = q8_gemv(x.clone(), weights[i % N_WEIGHTS].clone());
    });

    measure("C  NEW split-K, rotating 4 weights (DRAM)", &device, |i| {
        let _ = q8_gemv_partial_sum(x.clone(), weights[i % N_WEIGHTS].clone());
    });
}

fn measure<F: FnMut(usize)>(label: &str, device: &Device, mut call: F) {
    for i in 0..WARMUP {
        call(i);
    }
    device.sync().unwrap();

    let t0 = Instant::now();
    for i in 0..ITERS {
        call(i);
    }
    let enqueue = t0.elapsed();

    let t1 = Instant::now();
    device.sync().unwrap();
    let drain = t1.elapsed();

    let total = enqueue + drain;
    let per_call = |d: std::time::Duration| d.as_secs_f64() * 1e6 / ITERS as f64;
    let gbs = weight_bytes() * ITERS as f64 / total.as_secs_f64() / 1e9;

    println!("{label}");
    println!("  cpu enqueue  : {:8.1} us/call", per_call(enqueue));
    println!("  wall / call  : {:8.1} us", per_call(total));
    println!("  effective bw : {:8.1} GB/s", gbs);
    println!();
}
