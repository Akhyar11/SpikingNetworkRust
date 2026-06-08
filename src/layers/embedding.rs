use rand::Rng;
use rayon::prelude::*;
use crate::core::delta::applyEmbeddingDelta;

use crate::layers::base::{BaseLayer, Layer};

pub struct SpikingEmbedding {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Vec<f32>,
    pub base: BaseLayer,
    
    pub beta: Vec<f32>,
    pub threshold: Vec<f32>,
    pub potentials: Vec<f32>,
    pub last_potentials: Vec<f32>,
    pub last_spikes: Option<Vec<f32>>,
    
    // Cache array ID token untuk backpropagation
    pub cached_inputs: Option<Vec<f32>>,
}

impl SpikingEmbedding {
    pub fn new(input_dim: usize, output_dim: usize, learning_rate: f32, clip_min: f32, clip_max: f32) -> Self {
        let mut rng = rand::thread_rng();
        // Xavier/Glorot Initialization
        let limit = (6.0 / (input_dim as f32 + output_dim as f32)).sqrt();
        let scale_factor = (input_dim as f32).sqrt(); // Skalakan seperti di TypeScript
        let mut weights = vec![0.0; input_dim * output_dim];
        for w in weights.iter_mut() {
            *w = rng.gen_range(-limit..limit) * scale_factor;
        }

        let mut beta = vec![0.0; output_dim];
        let mut threshold = vec![0.0; output_dim];
        for i in 0..output_dim {
            let shift = rng.gen_range(2..8) as i32; // 2 to 7 (equivalent to Math.floor(2 + Math.random() * 6))
            beta[i] = 1.0 - (1.0 / (1 << shift) as f32);
            threshold[i] = rng.gen_range(0.01..0.1);
        }

        Self {
            input_dim,
            output_dim,
            weights,
            base: BaseLayer::new("SpikingEmbedding", learning_rate, clip_min, clip_max),
            beta,
            threshold,
            potentials: Vec::new(),
            last_potentials: Vec::new(),
            last_spikes: None,
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
    pub fn reset_state(&mut self) {
        for p in self.potentials.iter_mut() { *p = 0.0; }
        for lp in self.last_potentials.iter_mut() { *lp = 0.0; }
        self.cached_inputs = None;
        self.last_spikes = None;
    }

    /// Mengembalikan flat matrix spikes dengan bentuk `[batch_size, output_dim]`.
    pub fn forward(&mut self, inputs: &[f32]) -> Vec<f32> {
        let batch_size = inputs.len();
        
        // Ensure potentials buffer shape
        let required_size = batch_size * self.output_dim;
        if self.potentials.len() != required_size {
            self.potentials = vec![0.0; required_size];
            self.last_potentials = vec![0.0; required_size];
        }

        let mut dot_data = vec![0.0; required_size];

        // Simpan input untuk backward pass
        self.cached_inputs = Some(inputs.to_vec());

        dot_data.par_chunks_mut(self.output_dim)
            .enumerate()
            .for_each(|(b, out_row)| {
                let token_id = inputs[b] as i32;
                if token_id > 0 && token_id < self.input_dim as i32 {
                    let w_offset = (token_id as usize) * self.output_dim;
                    for d in 0..self.output_dim {
                        out_row[d] = self.weights[w_offset + d];
                    }
                }
            });

        let mut output_spikes = vec![0.0; required_size];
        
        crate::core::lifStep::lifStep(
            &mut self.potentials,
            &dot_data,
            &mut output_spikes,
            &mut self.last_potentials,
            &self.beta,
            &self.threshold
        );
        
        self.last_spikes = Some(output_spikes.clone());

        output_spikes
    }

    /// Backward pass:
    /// `error_signal`: Matriks gradien dari layer selanjutnya (shape: `[batch_size, output_dim]`)
    pub fn backward(&mut self, error_signal: &[f32], b_matrix: Option<&[f32]>) {
        if let Some(inputs) = &self.cached_inputs {
            let mut e_hidden = vec![0.0; error_signal.len()];
            if let Some(b_mat) = b_matrix {
                let batch_seq = error_signal.len() / self.output_dim;
                for i in 0..batch_seq {
                    for j in 0..self.output_dim {
                        let mut sum = 0.0;
                        for k in 0..self.output_dim {
                            sum += error_signal[i * self.output_dim + k] * b_mat[k * self.output_dim + j];
                        }
                        e_hidden[i * self.output_dim + j] = sum;
                    }
                }
            } else {
                e_hidden.copy_from_slice(error_signal);
            }

            let mut masked_err = e_hidden.to_vec();
            crate::core::surrogate::maskSurrogate(&mut masked_err, &self.last_potentials, &self.threshold, 1.0);
            
            // Kita panggil fungsi applyEmbeddingDelta murni dari core
            applyEmbeddingDelta(
                &mut self.weights,
                inputs,
                &masked_err,
                self.base.learning_rate,
                self.input_dim,
                self.output_dim,
                self.base.clip_min,
                self.base.clip_max
            );
        }
    }
}
