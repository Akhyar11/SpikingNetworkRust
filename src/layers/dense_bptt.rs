use rand::Rng;
use ndarray::{ArrayView2, Array2};

use crate::core::dotProductAddOnly::dot_product_add_only;
use crate::core::lifStep::lifStep;
use crate::core::surrogate::maskSurrogate;
use crate::core::delta::applyAddOnlyDelta;

use crate::layers::base::{BaseLayer, Layer};

pub struct SpikingDenseBPTT {
    pub units: usize,
    pub in_features: usize,
    pub use_bias: bool,
    pub base: BaseLayer,
    
    pub kernel: Vec<f32>,
    pub bias: Vec<f32>,
    pub beta: Vec<f32>,
    pub threshold: Vec<f32>,

    pub potentials: Vec<f32>,

    pub history_inputs: Vec<Vec<f32>>,
    pub history_potentials: Vec<Vec<f32>>,
    pub history_spikes: Vec<Vec<f32>>,
    pub max_time_steps: usize,
}

impl SpikingDenseBPTT {
    pub fn new(
        in_features: usize, 
        units: usize, 
        use_bias: bool, 
        clip_min: f32, 
        clip_max: f32,
        _beta_range: (f32, f32),
        threshold_range: (f32, f32)
    ) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let limit = (6.0 / (in_features as f32 + units as f32)).sqrt();
        
        let mut kernel = vec![0.0; in_features * units];
        for w in kernel.iter_mut() {
            *w = rng.gen_range(-limit..limit);
        }

        let bias = vec![0.0; units];
        
        let mut beta = vec![0.0; units];
        let mut threshold = vec![0.0; units];
        
        for i in 0..units {
            let shift = rng.gen_range(2..6) as i32;
            beta[i] = 1.0 - (1.0 / (2.0f32).powi(shift));
            threshold[i] = rng.gen_range(threshold_range.0 .. threshold_range.1);
        }

        Self {
            units,
            in_features,
            use_bias,
            base: BaseLayer::new("SpikingDenseBPTT", 0.01, clip_min, clip_max),
            kernel,
            bias,
            beta,
            threshold,
            potentials: Vec::new(),
            history_inputs: Vec::new(),
            history_potentials: Vec::new(),
            history_spikes: Vec::new(),
            max_time_steps: 0,
        }
    }

    pub fn reset_sequence(&mut self, batch_size: usize, time_steps: usize) {
        self.max_time_steps = time_steps;
        self.potentials = vec![0.0; batch_size * self.units];
        self.history_inputs = vec![vec![]; time_steps];
        self.history_potentials = vec![vec![]; time_steps];
        self.history_spikes = vec![vec![]; time_steps];
    }

    pub fn compute_step(&mut self, inputs: &[f32], t: usize) -> Vec<f32> {
        let batch_size = inputs.len() / self.in_features;
        self.history_inputs[t] = inputs.to_vec();

        let inputs_arr = ArrayView2::from_shape((batch_size, self.in_features), inputs).unwrap();
        let kernel_arr = ArrayView2::from_shape((self.in_features, self.units), &self.kernel).unwrap();
        
        let mut dot: Array2<f32> = dot_product_add_only(&inputs_arr, &kernel_arr);

        if self.use_bias {
            for b in 0..batch_size {
                let mut row = dot.row_mut(b);
                for u in 0..self.units {
                    row[u] += self.bias[u];
                }
            }
        }

        let mut out_spikes = vec![0.0; batch_size * self.units];
        let mut pot_at_t = vec![0.0; batch_size * self.units];

        let dot_slice = dot.as_slice().unwrap();
        
        lifStep(
            &mut self.potentials,
            dot_slice,
            &mut out_spikes,
            &mut pot_at_t,
            &self.beta,
            &self.threshold
        );

        self.history_potentials[t] = pot_at_t.clone();
        self.history_spikes[t] = out_spikes.clone();

        out_spikes
    }

    pub fn learn_through_time(&mut self, error_sequence: &[Vec<f32>], learning_rate: f32) -> Vec<Vec<f32>> {
        if self.max_time_steps == 0 || self.history_inputs[0].is_empty() {
            panic!("Belum ada data memory di history_inputs! Jalankan compute_step() dulu.");
        }

        let batch_size = error_sequence[0].len() / self.units;
        let mut temporal_error_data = vec![0.0; batch_size * self.units];
        let mut masked_error_data = vec![0.0; batch_size * self.units];
        let window_size = 1.0;
        
        let mut error_wrt_inputs = vec![vec![0.0; batch_size * self.in_features]; self.max_time_steps];

        for t in (0..self.max_time_steps).rev() {
            let current_error_data = &error_sequence[t];
            let p_data = &self.history_potentials[t];
            let input_data = &self.history_inputs[t];

            for i in 0..masked_error_data.len() {
                masked_error_data[i] = current_error_data[i] + temporal_error_data[i];
            }

            maskSurrogate(&mut masked_error_data, p_data, &self.threshold, window_size);

            // C: Terapkan Add-Only Delta Rule untuk mengupdate matriks kernel dan bias pada waktu `t`
            applyAddOnlyDelta(
                &mut self.kernel,
                &mut self.bias,
                input_data,
                &masked_error_data,
                learning_rate,
                batch_size,
                self.in_features,
                self.units,
                self.use_bias,
                self.base.clip_min,
                self.base.clip_max
            );

            for b in 0..batch_size {
                let offset = b * self.units;
                for i in 0..self.units {
                    let idx = offset + i;
                    temporal_error_data[idx] = masked_error_data[idx] * self.beta[i];
                }
            }

            for b in 0..batch_size {
                let out_offset = b * self.units;
                let in_offset = b * self.in_features;
                for in_d in 0..self.in_features {
                    let mut sum = 0.0;
                    let k_offset = in_d * self.units;
                    for out_d in 0..self.units {
                        sum += masked_error_data[out_offset + out_d] * self.kernel[k_offset + out_d];
                    }
                    error_wrt_inputs[t][in_offset + in_d] = sum;
                }
            }
        } // Penutup loop waktu t
        
        error_wrt_inputs
    }
}

impl Layer for SpikingDenseBPTT {
    fn get_base_config(&self) -> &BaseLayer {
        &self.base
    }

    fn get_base_config_mut(&mut self) -> &mut BaseLayer {
        &mut self.base
    }
    
    fn get_parameters(&self) -> Vec<(&str, &[f32])> {
        vec![
            ("kernel", &self.kernel),
            ("bias", &self.bias),
            ("beta", &self.beta),
            ("threshold", &self.threshold),
        ]
    }

    fn set_parameter(&mut self, name: &str, data: &[f32]) -> Result<(), String> {
        match name {
            "kernel" => {
                if data.len() != self.kernel.len() { return Err("Ukuran kernel tidak cocok".into()); }
                self.kernel.copy_from_slice(data);
                Ok(())
            },
            "bias" => {
                if data.len() != self.bias.len() { return Err("Ukuran bias tidak cocok".into()); }
                self.bias.copy_from_slice(data);
                Ok(())
            },
            "beta" => {
                if data.len() != self.beta.len() { return Err("Ukuran beta tidak cocok".into()); }
                self.beta.copy_from_slice(data);
                Ok(())
            },
            "threshold" => {
                if data.len() != self.threshold.len() { return Err("Ukuran threshold tidak cocok".into()); }
                self.threshold.copy_from_slice(data);
                Ok(())
            },
            _ => Err(format!("Parameter {} tidak ditemukan di SpikingDenseBPTT", name))
        }
    }

    fn count_params(&self) -> usize {
        self.kernel.len() + self.bias.len()
    }

    fn get_output_shape(&self) -> String {
        format!("[batch_size, {}]", self.units)
    }
}
