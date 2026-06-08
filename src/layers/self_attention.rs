use ndarray::ArrayView2;
use rand::Rng;

use crate::layers::base::{BaseLayer, Layer};
use crate::core::dotProductAddOnly::dot_product_add_only;
use crate::core::lifStep::lifStep;

pub struct SpikingSelfAttention {
    pub d_model: usize,
    pub sequence_length: usize,
    pub base: BaseLayer,
    
    pub kernel_q: Vec<f32>,
    pub kernel_k: Vec<f32>,
    pub kernel_v: Vec<f32>,
    
    pub beta_qkv: Vec<f32>,
    pub threshold_qkv: Vec<f32>,
    
    pub potentials_q: Vec<f32>,
    pub potentials_k: Vec<f32>,
    pub potentials_v: Vec<f32>,

    pub last_inputs: Option<Vec<f32>>,
}

impl SpikingSelfAttention {
    pub fn new(
        d_model: usize, 
        sequence_length: usize, 
        learning_rate: f32, 
        clip_min: f32, 
        clip_max: f32, 
        beta_range: (f32, f32), 
        threshold_range: (f32, f32)
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut kernel_q = vec![0.0; d_model * d_model];
        let mut kernel_k = vec![0.0; d_model * d_model];
        let mut kernel_v = vec![0.0; d_model * d_model];
        
        // Optimasi: Skalakan bobot awal agar neuron bisa 'spike' dengan mulus
        let scale = (d_model as f32).sqrt();
        let limit = (6.0 / (d_model as f32 * 2.0)).sqrt() * scale;
        
        for i in 0..kernel_q.len() {
            kernel_q[i] = rng.gen_range(-limit..limit);
            kernel_k[i] = rng.gen_range(-limit..limit);
            kernel_v[i] = rng.gen_range(-limit..limit);
        }

        let mut beta_qkv = vec![0.0; d_model];
        let mut threshold_qkv = vec![0.0; d_model];
        for i in 0..d_model {
            beta_qkv[i] = rng.gen_range(beta_range.0 .. beta_range.1);
            threshold_qkv[i] = rng.gen_range(threshold_range.0 .. threshold_range.1);
        }

        Self {
            d_model,
            sequence_length,
            base: BaseLayer::new("SpikingSelfAttention", learning_rate, clip_min, clip_max),
            kernel_q,
            kernel_k,
            kernel_v,
            beta_qkv,
            threshold_qkv,
            potentials_q: Vec::new(),
            potentials_k: Vec::new(),
            potentials_v: Vec::new(),
            last_inputs: None,
        }
    }

    pub fn reset_state(&mut self, batch_size: usize) {
        let size = batch_size * self.sequence_length * self.d_model;
        self.potentials_q = vec![0.0; size];
        self.potentials_k = vec![0.0; size];
        self.potentials_v = vec![0.0; size];
    }

    /// Linear Attention: Q * (K^T * V) untuk mengatasi O(N^2)
    pub fn forward(&mut self, inputs: &[f32], actual_lengths: &[usize]) -> Vec<f32> {
        let batch_seq = inputs.len() / self.d_model;
        let batch = batch_seq / self.sequence_length;
        
        if batch * self.sequence_length != batch_seq {
            panic!("Jumlah input tidak sesuai dengan kelipatan sequence_length!");
        }

        // ALWAYS reset state on every forward pass to prevent potentials leaking across batches
        self.reset_state(batch);

        self.last_inputs = Some(inputs.to_vec());

        // 1. Proyeksi Spasial Spiking
        let inputs_arr = ArrayView2::from_shape((batch_seq, self.d_model), inputs).unwrap();
        let kq_arr = ArrayView2::from_shape((self.d_model, self.d_model), &self.kernel_q).unwrap();
        let kk_arr = ArrayView2::from_shape((self.d_model, self.d_model), &self.kernel_k).unwrap();
        let kv_arr = ArrayView2::from_shape((self.d_model, self.d_model), &self.kernel_v).unwrap();

        let dot_q = dot_product_add_only(&inputs_arr, &kq_arr);
        let dot_k = dot_product_add_only(&inputs_arr, &kk_arr);
        let dot_v = dot_product_add_only(&inputs_arr, &kv_arr);

        // 2. Evaluasi Potensial Membran dan Ekstrak Spike (S_Q, S_K, S_V)
        let mut sq = vec![0.0; batch_seq * self.d_model];
        let mut sk = vec![0.0; batch_seq * self.d_model];
        let mut sv = vec![0.0; batch_seq * self.d_model];
        let mut dummy = vec![0.0; batch_seq * self.d_model]; // Temp array untuk potensial sesaat yang tidak direkam di memori state

        lifStep(&mut self.potentials_q, dot_q.as_slice().unwrap(), &mut sq, &mut dummy, &self.beta_qkv, &self.threshold_qkv);
        lifStep(&mut self.potentials_k, dot_k.as_slice().unwrap(), &mut sk, &mut dummy, &self.beta_qkv, &self.threshold_qkv);
        lifStep(&mut self.potentials_v, dot_v.as_slice().unwrap(), &mut sv, &mut dummy, &self.beta_qkv, &self.threshold_qkv);

        let mut out_data = vec![0.0; batch_seq * self.d_model];

        // 3. Linear Attention (Menghitung pola atensi tanpa matrik memori inter-neuron raksasa N x N)
        for b in 0..batch {
            let mut kv_matrix = vec![0.0; self.d_model * self.d_model];

            // A: K^T * V
            let len_f32 = if actual_lengths[b] == 0 { 1.0 } else { actual_lengths[b] as f32 };
            for i in 0..self.sequence_length {
                if i >= actual_lengths[b] { continue; }
                let base_idx = (b * self.sequence_length + i) * self.d_model;
                let mut non_zero_k = Vec::with_capacity(self.d_model);
                let mut non_zero_v = Vec::with_capacity(self.d_model);
                
                for d in 0..self.d_model {
                    if sk[base_idx + d] > 0.0 { non_zero_k.push(d); }
                    if sv[base_idx + d] > 0.0 { non_zero_v.push(d); }
                }

                for &d1 in &non_zero_k {
                    for &d2 in &non_zero_v {
                        kv_matrix[d1 * self.d_model + d2] += 1.0 / len_f32;
                    }
                }
            }

            // B: Q * (K^T * V)
            for i in 0..self.sequence_length {
                if i >= actual_lengths[b] { continue; }
                let base_idx = (b * self.sequence_length + i) * self.d_model;
                let mut non_zero_q = Vec::with_capacity(self.d_model);
                for d in 0..self.d_model {
                    if sq[base_idx + d] > 0.0 { non_zero_q.push(d); }
                }

                for &d1 in &non_zero_q {
                    for d2 in 0..self.d_model {
                        let val = kv_matrix[d1 * self.d_model + d2];
                        if val > 0.0 {
                            out_data[base_idx + d2] += val;
                        }
                    }
                }
            }
        }

        // Clip / Binarize
        for x in out_data.iter_mut() {
            if *x > 1.0 { *x = 1.0; }
        }

        out_data
    }

    pub fn learn_attention(&mut self, error_signal: &[f32]) {
        let inputs = self.last_inputs.as_ref().expect("Panggil forward() dulu!");
        let batch_seq = inputs.len() / self.d_model;
        let lr = self.base.learning_rate;
        let clip_min = self.base.clip_min;
        let clip_max = self.base.clip_max;

        for b in 0..batch_seq {
            let offset = b * self.d_model;
            for i in 0..self.d_model {
                let in_val = inputs[offset + i];
                if in_val > 0.0 {
                    let k_offset = i * self.d_model;
                    for d in 0..self.d_model {
                        // Residu dopamin agar dead neuron tetap hidup kembali secara bertahap
                        let dopamine = 0.00005;
                        let delta = (lr * error_signal[offset + d] * in_val) + dopamine;
                        
                        let mut nq = self.kernel_q[k_offset + d] + delta;
                        if nq > clip_max { nq = clip_max; } else if nq < clip_min { nq = clip_min; }
                        self.kernel_q[k_offset + d] = nq;

                        let mut nk = self.kernel_k[k_offset + d] + delta;
                        if nk > clip_max { nk = clip_max; } else if nk < clip_min { nk = clip_min; }
                        self.kernel_k[k_offset + d] = nk;

                        let mut nv = self.kernel_v[k_offset + d] + delta;
                        if nv > clip_max { nv = clip_max; } else if nv < clip_min { nv = clip_min; }
                        self.kernel_v[k_offset + d] = nv;
                    }
                }
            }
        }
    }
}

impl Layer for SpikingSelfAttention {
    fn get_base_config(&self) -> &BaseLayer { &self.base }
    fn get_base_config_mut(&mut self) -> &mut BaseLayer { &mut self.base }
    
    fn get_parameters(&self) -> Vec<(&str, &[f32])> {
        vec![
            ("kernel_q", &self.kernel_q),
            ("kernel_k", &self.kernel_k),
            ("kernel_v", &self.kernel_v)
        ]
    }

    fn set_parameter(&mut self, name: &str, data: &[f32]) -> Result<(), String> {
        match name {
            "kernel_q" => {
                if data.len() != self.kernel_q.len() { return Err("Ukuran kernel_q tidak cocok".into()); }
                self.kernel_q.copy_from_slice(data);
                Ok(())
            },
            "kernel_k" => {
                if data.len() != self.kernel_k.len() { return Err("Ukuran kernel_k tidak cocok".into()); }
                self.kernel_k.copy_from_slice(data);
                Ok(())
            },
            "kernel_v" => {
                if data.len() != self.kernel_v.len() { return Err("Ukuran kernel_v tidak cocok".into()); }
                self.kernel_v.copy_from_slice(data);
                Ok(())
            },
            _ => Err(format!("Parameter {} tidak ditemukan di SpikingSelfAttention", name))
        }
    }

    fn count_params(&self) -> usize {
        self.kernel_q.len() + self.kernel_k.len() + self.kernel_v.len()
    }

    fn get_output_shape(&self) -> String {
        format!("[batch_size * {}, {}]", self.sequence_length, self.d_model)
    }
}
