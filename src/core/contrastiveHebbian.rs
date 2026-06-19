
#[allow(non_snake_case)]
pub fn contrastiveHebbian(
    spikes: &[f32],
    err_data: &mut [f32],
    num_pairs: usize,
    sequence_length: usize,
    d_model: usize,
    margin: f32,
    actual_lengths: &[usize]
) -> f32 {
    let mut total_loss: f32 = 0.0;
    
    let total_samples = actual_lengths.len();
    let has_hard_negative = total_samples >= 3 * num_pairs;

    for i in 0..num_pairs {
        let q_offset = i * sequence_length * d_model;
        let p_offset = (num_pairs + i) * sequence_length * d_model;
        
        let neg_idx = (i + 1) % num_pairs;
        let n1_offset = (num_pairs + neg_idx) * sequence_length * d_model;
        
        let n2_offset = if has_hard_negative {
            (2 * num_pairs + i) * sequence_length * d_model
        } else {
            0
        };
        
        let q_len = actual_lengths[i];

        for s in 0..sequence_length {
            if s >= q_len {
                continue;
            }

            for d in 0..d_model {
                let idx_q = q_offset + s * d_model + d;
                let idx_p = p_offset + s * d_model + d;
                let idx_n1 = n1_offset + s * d_model + d;

                let q_s = spikes[idx_q];
                let p_s = spikes[idx_p];
                let n1_s = spikes[idx_n1];
                let n2_s = if has_hard_negative { spikes[n2_offset + s * d_model + d] } else { 0.0 };

                let pull = p_s - q_s;
                
                let push1 = q_s * n1_s * margin; 
                let push2 = if has_hard_negative { q_s * n2_s * margin * 0.2 } else { 0.0 };
                
                let total_push = push1 + push2;

                if pull != 0.0 || total_push != 0.0 {
                    err_data[idx_q] += pull - total_push;
                    err_data[idx_p] += -pull;
                    err_data[idx_n1] += -push1;
                    if has_hard_negative {
                        let idx_n2 = n2_offset + s * d_model + d;
                        err_data[idx_n2] += -push2;
                    }
                    total_loss += pull.abs() + total_push;
                }
            }
        }
    }

    total_loss
}

#[allow(non_snake_case)]
pub fn distillationHebbian(
    spikes: &[f32],
    err_data: &mut [f32],
    num_pairs: usize,
    sequence_length: usize,
    d_model: usize,
    margin: f32,
    actual_lengths: &[usize],
    target_scores: &[f32]
) -> f32 {
    let mut total_loss: f32 = 0.0;
    
    for i in 0..num_pairs {
        let a_offset = (2 * i) * sequence_length * d_model;
        let b_offset = (2 * i + 1) * sequence_length * d_model;
        
        let target_score = target_scores[i].clamp(0.0, 1.0);
        
        let a_len = actual_lengths[2 * i];
        let b_len = actual_lengths[2 * i + 1];
        let max_len = a_len.max(b_len);

        for s in 0..sequence_length {
            if s >= max_len { continue; }

            for d in 0..d_model {
                let idx_a = a_offset + s * d_model + d;
                let idx_b = b_offset + s * d_model + d;

                let a_s = spikes[idx_a];
                let b_s = spikes[idx_b];

                let shared_active = if (a_s > 0.0 && b_s > 0.0) || (a_s == 0.0 && b_s == 0.0) { 1.0 } else { 0.0 };
                let error = target_score - shared_active;

                if error > 0.0 {
                    if a_s > 0.0 && b_s == 0.0 {
                        err_data[idx_b] += error * margin;
                        total_loss += error * margin;
                    } else if a_s == 0.0 && b_s > 0.0 {
                        err_data[idx_a] += error * margin;
                        total_loss += error * margin;
                    }
                } else if error < 0.0 {
                    if a_s > 0.0 && b_s > 0.0 {
                        err_data[idx_a] += error * margin; 
                        err_data[idx_b] += error * margin;
                        total_loss += (-error) * margin;
                    }
                }
            }
        }
    }

    total_loss
}

#[allow(non_snake_case)]
pub fn poolerDistillation(
    normalized_out: &[f32],
    err_data: &mut [f32],
    num_pairs: usize,
    d_model: usize,
    margin: f32,
    target_scores: &[f32]
) -> f32 {
    let mut total_loss: f32 = 0.0;
    for i in 0..num_pairs {
        let a_offset = (2 * i) * d_model;
        let b_offset = (2 * i + 1) * d_model;
        let target = target_scores[i];

        // Compute cosine similarity (already normalized, so it's just dot product)
        let mut sim = 0.0;
        for d in 0..d_model {
            sim += normalized_out[a_offset + d] * normalized_out[b_offset + d];
        }

        let error = target - sim; // positive if sim is too low
        total_loss += error.abs() * margin;

        for d in 0..d_model {
            let a_val = normalized_out[a_offset + d];
            let b_val = normalized_out[b_offset + d];
            
            // Gradient of sim w.r.t a_val is b_val
            // Gradient of sim w.r.t b_val is a_val
            // We want to maximize sim if error > 0, minimize if error < 0
            err_data[a_offset + d] += error * b_val * margin;
            err_data[b_offset + d] += error * a_val * margin;
        }
    }
    total_loss
}
