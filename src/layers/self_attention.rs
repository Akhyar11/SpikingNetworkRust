use ndarray::ArrayView2;
use rand::Rng;

use crate::layers::base::{BaseLayer, Layer};
use crate::core::dotProduct::dot_product;
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
    
    pub beta_scores: Vec<f32>,
    pub threshold_scores: Vec<f32>,
    pub potentials_scores: Vec<f32>,

    pub last_inputs: Option<Vec<f32>>,
}

impl SpikingSelfAttention {
    pub fn new(
        d_model: usize, 
        sequence_length: usize, 
        learning_rate: f32, 
        clip_min: f32, 
        clip_max: f32, 
        _beta_range: (f32, f32), 
        threshold_range: (f32, f32)
    ) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut kernel_q = vec![0.0; d_model * d_model];
        let mut kernel_k = vec![0.0; d_model * d_model];
        let mut kernel_v = vec![0.0; d_model * d_model];
        
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
            let shift = rng.gen_range(2..8) as i32;
            beta_qkv[i] = 1.0 - (1.0 / (1 << shift) as f32);
            threshold_qkv[i] = rng.gen_range(threshold_range.0 .. threshold_range.1);
        }

        let mut beta_scores = vec![0.0; sequence_length];
        let mut threshold_scores = vec![0.0; sequence_length];
        for i in 0..sequence_length {
            beta_scores[i] = 0.9;
            threshold_scores[i] = 1.0;
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
            beta_scores,
            threshold_scores,
            potentials_scores: Vec::new(),
            last_inputs: None,
        }
    }

    pub fn reset_state(&mut self, batch_size: usize) {
        let size = batch_size * self.sequence_length * self.d_model;
        self.potentials_q = vec![0.0; size];
        self.potentials_k = vec![0.0; size];
        self.potentials_v = vec![0.0; size];
        self.potentials_scores = vec![0.0; batch_size * self.sequence_length * self.sequence_length];
    }

    pub fn forward(&mut self, inputs: &[f32], actual_lengths: &[usize]) -> Vec<f32> {
        let batch_seq = inputs.len() / self.d_model;
        let batch = batch_seq / self.sequence_length;
        
        if batch * self.sequence_length != batch_seq {
            panic!("Jumlah input tidak sesuai dengan kelipatan sequence_length!");
        }

        self.reset_state(batch);

        self.last_inputs = Some(inputs.to_vec());

        let inputs_arr = ArrayView2::from_shape((batch_seq, self.d_model), inputs).unwrap();
        let kq_arr = ArrayView2::from_shape((self.d_model, self.d_model), &self.kernel_q).unwrap();
        let kk_arr = ArrayView2::from_shape((self.d_model, self.d_model), &self.kernel_k).unwrap();
        let kv_arr = ArrayView2::from_shape((self.d_model, self.d_model), &self.kernel_v).unwrap();

        let dot_q = dot_product(&inputs_arr, &kq_arr);
        let dot_k = dot_product(&inputs_arr, &kk_arr);
        let dot_v = dot_product(&inputs_arr, &kv_arr);

        let mut sq = vec![0.0; batch_seq * self.d_model];
        let mut sk = vec![0.0; batch_seq * self.d_model];
        let mut sv = vec![0.0; batch_seq * self.d_model];
        let mut dummy = vec![0.0; batch_seq * self.d_model];

        lifStep(&mut self.potentials_q, dot_q.as_slice().unwrap(), &mut sq, &mut dummy, &self.beta_qkv, &self.threshold_qkv);
        lifStep(&mut self.potentials_k, dot_k.as_slice().unwrap(), &mut sk, &mut dummy, &self.beta_qkv, &self.threshold_qkv);
        lifStep(&mut self.potentials_v, dot_v.as_slice().unwrap(), &mut sv, &mut dummy, &self.beta_qkv, &self.threshold_qkv);

        let _out_data = vec![0.0; batch_seq * self.d_model];

        let mut match_scores = vec![0.0; batch * self.sequence_length * self.sequence_length];
        
        for b in 0..batch {
            for i in 0..self.sequence_length {
                if i >= actual_lengths[b] { continue; }
                let q_base = b * self.sequence_length * self.d_model + i * self.d_model;
                
                let mut non_zero_q = Vec::with_capacity(self.d_model);
                for d in 0..self.d_model {
                    if sq[q_base + d] > 0.0 { non_zero_q.push(d); }
                }
                if non_zero_q.is_empty() { continue; }

                let mut max_match = 0;
                let mut temp_matches = vec![0; self.sequence_length];
                
                for j in 0..self.sequence_length {
                    if j >= actual_lengths[b] { continue; }
                    let mut match_count = 0;
                    let k_base = b * self.sequence_length * self.d_model + j * self.d_model;
                    for &d in &non_zero_q {
                        if sk[k_base + d] > 0.0 { match_count += 1; }
                    }
                    temp_matches[j] = match_count;
                    if match_count > max_match {
                        max_match = match_count;
                    }
                }
                
                for j in 0..self.sequence_length {
                    let score_idx = b * self.sequence_length * self.sequence_length + i * self.sequence_length + j;
                    if max_match > 0 {
                        match_scores[score_idx] = temp_matches[j] as f32 / max_match as f32;
                    } else {
                        match_scores[score_idx] = 0.0;
                    }
                }
            }
        }

        let mut s_scores_data = vec![0.0; batch * self.sequence_length * self.sequence_length];
        let mut dummy_lp_scores = vec![0.0; batch * self.sequence_length * self.sequence_length];
        lifStep(&mut self.potentials_scores, &match_scores, &mut s_scores_data, &mut dummy_lp_scores, &self.beta_scores, &self.threshold_scores);

        let mut out_data = vec![0.0; batch_seq * self.d_model];

        for b in 0..batch {
            for j in 0..self.sequence_length {
                if j >= actual_lengths[b] { continue; }
                let v_base = b * self.sequence_length * self.d_model + j * self.d_model;
                let mut non_zero_v = Vec::with_capacity(self.d_model);
                for d in 0..self.d_model {
                    if sv[v_base + d] > 0.0 { non_zero_v.push(d); }
                }
                if non_zero_v.is_empty() { continue; }

                for i in 0..self.sequence_length {
                    if i >= actual_lengths[b] { continue; }
                    let score_idx = b * self.sequence_length * self.sequence_length + i * self.sequence_length + j;
                    let graded_score = s_scores_data[score_idx];
                    if graded_score > 0.0 {
                        let out_base = b * self.sequence_length * self.d_model + i * self.d_model;
                        for &d in &non_zero_v {
                            out_data[out_base + d] += graded_score * sv[v_base + d];
                        }
                    }
                }
            }
        }

        for x in out_data.iter_mut() {
            if *x > 1.0 { *x = 1.0; }
        }

        out_data
    }

    pub fn learn_attention(&mut self, error_signal: &[f32], actual_lengths: &[usize]) {
        let inputs = self.last_inputs.as_ref().expect("Panggil forward() dulu!");
        let _batch_seq = inputs.len() / self.d_model;
        let batch_size = actual_lengths.len();
        let lr = self.base.learning_rate;
        let clip_min = self.base.clip_min;
        let clip_max = self.base.clip_max;

        for b in 0..batch_size {
            for t in 0..self.sequence_length {
                if t >= actual_lengths[b] { continue; }
                
                let b_seq = b * self.sequence_length + t;
                let offset = b_seq * self.d_model;
                for i in 0..self.d_model {
                    let in_val = inputs[offset + i];
                    if in_val > 0.0 {
                        let k_offset = i * self.d_model;
                        for d in 0..self.d_model {
                            let dopamine = 0.0;
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
}

impl Layer for SpikingSelfAttention {
    fn get_base_config(&self) -> &BaseLayer { &self.base }
    fn get_base_config_mut(&mut self) -> &mut BaseLayer { &mut self.base }
    
    fn get_parameters(&self) -> Vec<(&str, &[f32])> {
        vec![
            ("kernel_q", &self.kernel_q),
            ("kernel_k", &self.kernel_k),
            ("kernel_v", &self.kernel_v),
            ("beta_qkv", &self.beta_qkv),
            ("threshold_qkv", &self.threshold_qkv),
            ("beta_scores", &self.beta_scores),
            ("threshold_scores", &self.threshold_scores),
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
            "beta_qkv" => {
                if data.len() != self.beta_qkv.len() { return Err("Ukuran beta_qkv tidak cocok".into()); }
                self.beta_qkv.copy_from_slice(data);
                Ok(())
            },
            "threshold_qkv" => {
                if data.len() != self.threshold_qkv.len() { return Err("Ukuran threshold_qkv tidak cocok".into()); }
                self.threshold_qkv.copy_from_slice(data);
                Ok(())
            },
            "beta_scores" => {
                if data.len() != self.beta_scores.len() { return Err("Ukuran beta_scores tidak cocok".into()); }
                self.beta_scores.copy_from_slice(data);
                Ok(())
            },
            "threshold_scores" => {
                if data.len() != self.threshold_scores.len() { return Err("Ukuran threshold_scores tidak cocok".into()); }
                self.threshold_scores.copy_from_slice(data);
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
