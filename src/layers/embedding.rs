use rand::Rng;
use rayon::prelude::*;
use crate::core::delta::applyEmbeddingDelta;

pub struct SpikingEmbedding {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Vec<f32>,
    pub learning_rate: f32,
    
    // Cache array ID token untuk backpropagation
    cached_inputs: Option<Vec<f32>>,
}

impl SpikingEmbedding {
    pub fn new(input_dim: usize, output_dim: usize, learning_rate: f32) -> Self {
        let mut rng = rand::thread_rng();
        // Xavier/Glorot Initialization
        let limit = (6.0 / (input_dim as f32 + output_dim as f32)).sqrt();
        let mut weights = vec![0.0; input_dim * output_dim];
        for w in weights.iter_mut() {
            *w = rng.gen_range(-limit..limit);
        }

        Self {
            input_dim,
            output_dim,
            weights,
            learning_rate,
            cached_inputs: None,
        }
    }

    /// Forward pass:
    /// `inputs`: Array 1D yang berisi urutan Token ID.
    /// Mengembalikan flat matrix spikes dengan bentuk `[batch_size, output_dim]`.
    pub fn forward(&mut self, inputs: &[f32]) -> Vec<f32> {
        let batch_size = inputs.len();
        let mut output_spikes = vec![0.0; batch_size * self.output_dim];

        // Simpan input untuk backward pass
        self.cached_inputs = Some(inputs.to_vec());

        output_spikes.par_chunks_mut(self.output_dim)
            .enumerate()
            .for_each(|(b, out_row)| {
                let token_id = inputs[b] as i32;
                // Hanya proses jika Token ID valid (melewati <PAD> yang biasanya = 1 atau <=0)
                if token_id > 0 && token_id < self.input_dim as i32 {
                    let w_offset = (token_id as usize) * self.output_dim;
                    for d in 0..self.output_dim {
                        let weight_val = self.weights[w_offset + d];
                        // ==========================================
                        // STATELESS BINARIZATION (Heaviside Step)
                        // Bobot positif = Spike (1.0), Negatif = Mati (0.0)
                        // ==========================================
                        out_row[d] = if weight_val > 0.0 { 1.0 } else { 0.0 };
                    }
                }
            });

        output_spikes
    }

    /// Backward pass:
    /// `error_signal`: Matriks gradien dari layer selanjutnya (shape: `[batch_size, output_dim]`)
    pub fn backward(&mut self, error_signal: &[f32]) {
        if let Some(inputs) = &self.cached_inputs {
            // Kita panggil fungsi applyEmbeddingDelta murni dari core
            applyEmbeddingDelta(
                &mut self.weights,
                inputs,
                error_signal,
                self.learning_rate,
                self.input_dim,
                self.output_dim
            );
        }
    }
}
