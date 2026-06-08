use rayon::prelude::*;

/// Kalkulasi Fungsi Loss "In-Batch Negative Contrastive" khusus SNN.
/// Menarik vektor Q ke arah Positif (P) dan menjauhkannya dari Negatif (N).
#[allow(non_snake_case)]
pub fn contrastiveHebbian(
    spikes: &[f32],
    err_data: &mut [f32],
    num_pairs: usize,
    sequence_length: usize,
    d_model: usize,
    margin: f32
) -> f32 {
    let mut total_loss: f32 = 0.0;

    // Kita asumsikan `spikes` ini sebenarnya adalah float embedding L2-Normalized dari tahapan sebelumnya.
    // Jika mereka adalah L2 Normalized, ||q-p||^2 = 2 - 2(q . p).
    // Jadi Loss = max(0, (q.n) - (q.p) + margin)
    
    // Karena butuh state penuh, lebih aman diserialkan di sini untuk memastikan thread-safety.
    // Rayon masih bisa dipakai untuk pair_losses, namun kita simpan iterasi untuk mutasi.
    
    for i in 0..num_pairs {
        let q_offset = i * sequence_length * d_model;
        let p_offset = (num_pairs + i) * sequence_length * d_model;
        let neg_idx = (i + 1) % num_pairs;
        let n_offset = (num_pairs + neg_idx) * sequence_length * d_model;

        // 1. Hitung Dot Product
        let mut dot_qp = 0.0;
        let mut dot_qn = 0.0;
        for s in 0..sequence_length {
            for d in 0..d_model {
                let q_s = spikes[q_offset + s * d_model + d];
                let p_s = spikes[p_offset + s * d_model + d];
                let n_s = spikes[n_offset + s * d_model + d];
                dot_qp += q_s * p_s;
                dot_qn += q_s * n_s;
            }
        }

        // 2. Hitung Loss Margin
        let loss = (dot_qn - dot_qp + margin).max(0.0);
        total_loss += loss;

        // 3. Backpropagate jika loss aktif
        if loss > 0.0 {
            for s in 0..sequence_length {
                for d in 0..d_model {
                    let idx_q = q_offset + s * d_model + d;
                    let idx_p = p_offset + s * d_model + d;
                    let idx_n = n_offset + s * d_model + d;

                    let q_s = spikes[idx_q];
                    let p_s = spikes[idx_p];
                    let n_s = spikes[idx_n];

                    // Gradient:
                    // dL/dq = n - p  => -dL/dq = p - n
                    // dL/dp = -q     => -dL/dp = q
                    // dL/dn = q      => -dL/dn = -q
                    
                    let mut grad_q = p_s - n_s;
                    let mut grad_p = q_s;
                    let mut grad_n = -q_s;

                    // SYMMETRY BREAKING: Jika embeddings kolaps (identik), berikan dorongan acak deterministik
                    // agar tidak stuck di gradien nol.
                    if grad_q.abs() < 1e-5 && (q_s - p_s).abs() < 1e-5 {
                        let noise = if (d + i) % 2 == 0 { 0.05 } else { -0.05 };
                        grad_q = noise;
                        grad_p = noise;
                        grad_n = -noise;
                    }
                    
                    err_data[idx_q] += grad_q;
                    err_data[idx_p] += grad_p;
                    err_data[idx_n] += grad_n; 
                }
            }
        }
    }

    total_loss
}
