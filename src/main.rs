use std::{io::Write, time::Instant};

use burn::{
    Tensor,
    module::{Module, Quantizer},
    tensor::{
        DType, Device, Int,
        quantization::{Calibration, QuantScheme, ScaleDtype},
    },
};
use burn_store::{
    FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use qwen3_5::{model::TextForCausalLMConfig, text_config::get_text_config, text_modules::Cache};
use tokenizers::Tokenizer;

fn main() {
    // _quantize_and_save();
    chat();
}

/// Commands:
/// /reset => fresh conversation
/// /exit  => quit
fn chat() {
    let device = Device::wgpu(Default::default());
    println!("Initializing Qwen 3.5 Text Model (chat)");
    let max_output_tokens = 1024;

    let config = get_text_config();
    let model = TextForCausalLMConfig::new().init(&config, &device);
    let model = model.load_file("./model.bpk");

    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file("tokenizer.json").expect("Failed to load tokenizer.json");

    let im_end_id: u32 = tokenizer
        .token_to_id("<|im_end|>")
        .expect("tokenizer must contain <|im_end|>");
    let eos_id = config.eos_token_id as u32;

    println!("Qwen 3.5 chat - up to {max_output_tokens} new tokens per reply.");
    println!("Commands: /reset (new conversation), /exit (quit).");

    let system = "You are a helpful assistant.";
    let mut cache = Cache::new(config.num_hidden_layers);
    let mut turns = 0usize;
    let mut total_tokens = 0usize;

    loop {
        let mut line = String::new();
        print!("\nYou > ");
        std::io::stdout().flush().unwrap();
        if std::io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
            break; // EOF
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        match text {
            "/exit" | "/quit" | "exit" | "quit" => break,
            "/reset" => {
                cache = Cache::new(config.num_hidden_layers);
                turns = 0;
                total_tokens = 0;
                println!("(conversation reset)");
                continue;
            }
            _ => {}
        }

        let turn = if turns == 0 {
            format!(
                "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            )
        } else {
            format!("\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
        };

        let encoding = tokenizer.encode(turn.as_str(), false).unwrap();
        let ids_i32: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
        let seq_len = ids_i32.len();
        total_tokens += seq_len;

        let input_ids =
            Tensor::<1, Int>::from_ints(ids_i32.as_slice(), &device).unsqueeze_dim::<2>(0);
        let logits = model.forward(input_ids, None, None, Some(&mut cache));

        let prefill_argmax = logits.argmax(2);
        let mut next_token = prefill_argmax
            .slice([0..1, (seq_len - 1)..seq_len, 0..1])
            .reshape([1, 1]);

        print!("Qwen: ");
        std::io::stdout().flush().unwrap();

        let gen_start = Instant::now();
        let mut produced = 0usize;
        let mut generated: Vec<u32> = Vec::new();
        let mut printed_chars = 0usize;

        for _ in 0..max_output_tokens {
            let token_id: u32 =
                next_token.clone().into_data().try_to_vec::<i32>().unwrap()[0] as u32;
            if token_id == im_end_id || token_id == eos_id {
                break;
            }
            generated.push(token_id);

            let decoded = tokenizer.decode(&generated, true).unwrap();
            let mut new_text: String = decoded.chars().skip(printed_chars).collect();
            if new_text.ends_with('\u{FFFD}') {
                new_text.pop();
                printed_chars = decoded.chars().count() - 1;
            } else {
                printed_chars = decoded.chars().count();
            }
            if !new_text.is_empty() {
                print!("{new_text}");
                std::io::stdout().flush().unwrap();
            }
            produced += 1;
            total_tokens += 1;

            let logits = model.forward(next_token.clone(), None, None, Some(&mut cache));
            next_token = logits.argmax(2).reshape([1, 1]);
        }

        let tail: String = tokenizer
            .decode(&generated, true)
            .unwrap()
            .chars()
            .skip(printed_chars)
            .collect();
        print!("{tail}");
        std::io::stdout().flush().unwrap();

        let stop = [im_end_id as i32];
        let im_end_t = Tensor::<1, Int>::from_ints(stop.as_slice(), &device).unsqueeze_dim::<2>(0);
        model.forward(im_end_t, None, None, Some(&mut cache));
        total_tokens += 1;

        println!();
        let secs = gen_start.elapsed().as_secs_f64();
        let speed = if secs > 0.0 {
            produced as f64 / secs
        } else {
            0.0
        };
        println!("[{produced} tok, {speed:.1} tok/s, ctx ~{total_tokens} tok]");
        turns += 1;
    }
}

fn _quantize_and_save() {
    let device = Device::wgpu(Default::default());
    let config = get_text_config();
    let mut model = TextForCausalLMConfig::new().init(&config, &device);

    let make_store = |pattern: &str| {
        let adapter = PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32));
        SafetensorsStore::from_file("./model.safetensors-00001-of-00001.safetensors")
            .with_from_adapter(adapter)
            .with_key_remapping(r"^model\.", "")
            .with_key_remapping(r"\.A_log", ".a_log")
            .with_regex(pattern)
            .allow_partial(true)
    };

    let scheme = QuantScheme::default().per_block([64], ScaleDtype::F32);
    let mut quantizer = Quantizer::new(Calibration::MinMax, scheme);

    let mut store = make_store(r"^(model\.)?language_model\.(embed_tokens|norm)\.");
    let result = model.load_from(&mut store).unwrap();
    println!("embed+norm: applied {}", result.applied.len());

    // Quantized using per-tensor scale for ease of learning
    let embed_scheme = QuantScheme::default();
    let mut embed_quantizer = Quantizer::new(Calibration::MinMax, embed_scheme);

    model.language_model.embed_tokens = model
        .language_model
        .embed_tokens
        .quantize_weights(&mut embed_quantizer);

    device.sync().unwrap();

    for i in 0..config.num_hidden_layers {
        let mut store = make_store(&format!(r"^(model\.)?language_model\.layers\.{i}\."));
        let result = model.load_from(&mut store).unwrap();
        println!("layer {i}: applied {}", result.applied.len());

        let layer = &mut model.language_model.layers[i];
        layer.mlp = layer.mlp.clone().quantize_weights(&mut quantizer);
        if let Some(self_attn) = &mut layer.self_attn {
            self_attn.q_proj = self_attn.q_proj.clone().quantize_weights(&mut quantizer);
            self_attn.k_proj = self_attn.k_proj.clone().quantize_weights(&mut quantizer);
            self_attn.v_proj = self_attn.v_proj.clone().quantize_weights(&mut quantizer);
            self_attn.o_proj = self_attn.o_proj.clone().quantize_weights(&mut quantizer);
        }
        if let Some(linear_attn) = &mut layer.linear_attn {
            linear_attn.in_proj_qkv = linear_attn
                .in_proj_qkv
                .clone()
                .quantize_weights(&mut quantizer);
            linear_attn.out_proj = linear_attn
                .out_proj
                .clone()
                .quantize_weights(&mut quantizer);
            linear_attn.in_proj_z = linear_attn
                .in_proj_z
                .clone()
                .quantize_weights(&mut quantizer);
        }
        device.sync().unwrap();
    }

    model.save_file("./model.bpk").unwrap();
}
