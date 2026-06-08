use rand::Rng;
use rayon::prelude::*;
use crate::core::delta::applyEmbeddingDelta;

use crate::layers::base::{BaseLayer, Layer};

pub struct SpikingEmbedding {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Vec<f32>,
    pub base: BaseLayer,
    
    // Cache array ID token untuk backpropagation
    cached_inputs: Option<Vec<f32>>,
}

impl SpikingEmbedding {
    pub fn new(input_dim: usize, output_dim: usize, learning_rate: f32, clip_min: f32, clip_max: f32) -> Self {
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
            base: BaseLayer::new("SpikingEmbedding", learning_rate, clip_min, clip_max),
            cached_inputs: None,
        }
    }
}

impl Layer for SpikingEmbedding {
    fn get_base_config(&self) -> &BaseLayer {
        &self.base
    }

    fn get_base_config_mut(&mut self) -> &mut BaseLayer {
        &mut self.base
    }
    
    fn get_parameters(&self) -> Vec<(&str, &[f32])> {
        vec![("weights", &self.weights)]
    }

    fn set_parameter(&mut self, name: &str, data: &[f32]) -> Result<(), String> {
        if name == "weights" {
            if data.len() != self.weights.len() {
                return Err("Ukuran bobot (shape) tidak cocok saat memuat model".to_string());
            }
            self.weights.copy_from_slice(data);
            Ok(())
        } else {
            Err(format!("Parameter {} tidak ditemukan di SpikingEmbedding", name))
        }
    }

    fn count_params(&self) -> usize {
        self.weights.len()
    }

    fn get_output_shape(&self) -> String {
        format!("[batch_size, {}]", self.output_dim)
    }
}

impl SpikingEmbedding {
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
                self.cached_inputs.as_ref().unwrap(),
                error_signal,
                self.base.learning_rate,
                self.input_dim,
                self.output_dim,
                self.base.clip_min,
                self.base.clip_max
            );
        }
    }
}
