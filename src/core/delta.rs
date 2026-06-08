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
    use_bias: bool
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
            // Clipping sederhana agar bobot tetap sehat
            if *w > 1.0 { *w = 1.0; }
            if *w < -1.0 { *w = -1.0; }
        });

    if use_bias {
        bias.par_iter_mut()
            .zip(bias_gradients.par_iter())
            .for_each(|(b, &g)| {
                *b += learning_rate * g;
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
    output_dim: usize
) {
    let batch = inputs.len(); 

    for b in 0..batch {
        let token_idx = inputs[b] as i32;
        
        // Lewati padding token atau OOV yang tidak valid
        if token_idx > 0 && token_idx < input_dim as i32 {
            let token_idx = token_idx as usize;
            let emb_offset = token_idx * output_dim;
            let err_offset = b * output_dim;

            for out_d in 0..output_dim {
                let err = error_signal[err_offset + out_d];
                let mut new_w = embeddings[emb_offset + out_d] + (learning_rate * err);
                
                // Clip weights
                if new_w > 1.0 { new_w = 1.0; }
                if new_w < -1.0 { new_w = -1.0; }
                
                embeddings[emb_offset + out_d] = new_w;
            }
        }
    }
}
