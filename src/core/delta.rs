use std::collections::HashMap;
use rayon::prelude::*;

/// Mengaplikasikan gradien ke kernel bobot untuk Dense layer (Add-Only)
#[allow(non_snake_case)]
pub fn applyAddOnlyDelta(
    kernel: &mut [f32],
    bias: &mut [f32],
    inputs: &[f32],
    error_signal: &[f32],
    learning_rate: f32,
    batch: usize,
    in_features: usize,
    units: usize,
    use_bias: bool,
    clip_min: f32,
    clip_max: f32
) {
    let mut weight_gradients = vec![0.0; in_features * units];
    let mut bias_gradients = vec![0.0; units];

    // Hitung akumulasi gradien
    for b in 0..batch {
        let in_offset = b * in_features;
        let err_offset = b * units;

        for out_d in 0..units {
            let err = error_signal[err_offset + out_d];
            if err != 0.0 {
                // Untuk bias
                if use_bias {
                    bias_gradients[out_d] += err;
                }
                
                // Untuk bobot (ingat inputnya SNN, sebagian besar biner)
                for in_d in 0..in_features {
                    let inp = inputs[in_offset + in_d];
                    // Add-Only Gradient: gradien hanya berefek jika presinaptik neuron pernah menyala (1.0)
                    if inp > 0.0 {
                        weight_gradients[in_d * units + out_d] += inp * err;
                    }
                }
            }
        }
    }

    // Aplikasikan gradien ke memori aslinya menggunakan Rayon
    kernel.par_iter_mut()
        .zip(weight_gradients.par_iter())
        .for_each(|(w, &g)| {
            *w += learning_rate * g;
            // Clipping sesuai parameter
            if *w > clip_max { *w = clip_max; }
            if *w < clip_min { *w = clip_min; }
        });

    if use_bias {
        bias.par_iter_mut()
            .zip(bias_gradients.par_iter())
            .for_each(|(b, &g)| {
                *b += learning_rate * g;
                if *b > clip_max { *b = clip_max; }
                if *b < clip_min { *b = clip_min; }
            });
    }
}

/// Mengaplikasikan gradien ke Embedding Matrix
/// Asumsinya `inputs` adalah array dari Token IDs (1D), bukan matriks one-hot.
#[allow(non_snake_case)]
pub fn applyEmbeddingDelta(
    embeddings: &mut [f32],
    inputs: &[f32],
    error_signal: &[f32],
    learning_rate: f32,
    input_dim: usize,
    output_dim: usize,
    clip_min: f32,
    clip_max: f32
) {
    let batch = inputs.len(); 
    let mut grad_accum: HashMap<usize, Vec<f32>> = HashMap::new();

    // Akumulasi gradien per token (menggabungkan token yang muncul berulang kali di batch)
    for b in 0..batch {
        let token_idx = inputs[b] as i32;
        
        // Lewati padding token atau OOV yang tidak valid
        if token_idx > 0 && token_idx < input_dim as i32 {
            let token_idx = token_idx as usize;
            let err_offset = b * output_dim;

            let entry = grad_accum.entry(token_idx).or_insert_with(|| vec![0.0; output_dim]);
            for out_d in 0..output_dim {
                entry[out_d] += error_signal[err_offset + out_d];
            }
        }
    }

    // Terapkan gradien yang terakumulasi HANYA SEKALI ke memori asli
    for (token_idx, acc_err) in grad_accum {
        let emb_offset = token_idx * output_dim;
        for out_d in 0..output_dim {
            let mut new_w = embeddings[emb_offset + out_d] + (learning_rate * acc_err[out_d]);
            
            // Clip weights sesuai batasan fleksibel
            if new_w > clip_max { new_w = clip_max; }
            if new_w < clip_min { new_w = clip_min; }
            
            embeddings[emb_offset + out_d] = new_w;
        }
    }
}
