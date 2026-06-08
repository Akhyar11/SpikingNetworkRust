use rayon::prelude::*;

/// Kalkulasi Fungsi Loss "In-Batch Negative Contrastive" khusus SNN.
/// Menarik vektor Q ke arah Positif (P) dan menjauhkannya dari Negatif (N).
#[allow(non_snake_case)]
pub fn contrastiveHebbian(
    spikes: &[f32],
    err_data: &mut [f32],
    num_pairs: usize,
    sequence_length: usize,
    d_model: usize
) -> f32 {
    let mut total_loss: f32 = 0.0;

    let pair_losses: Vec<f32> = (0..num_pairs).into_par_iter().map(|i| {
        let mut local_loss = 0.0;
        let q_offset = i * sequence_length * d_model;
        let p_offset = (num_pairs + i) * sequence_length * d_model;
        
        // Sampling negatif in-batch: kita gunakan tetangga terdekat secara berputar
        let neg_idx = (i + 1) % num_pairs;
        let n_offset = (num_pairs + neg_idx) * sequence_length * d_model;

        // Gunakan pointer offset secara aman namun harus disimulasikan menggunakan indexing yang efisien
        for s in 0..sequence_length {
            for d in 0..d_model {
                let idx_q = q_offset + s * d_model + d;
                let idx_p = p_offset + s * d_model + d;
                let idx_n = n_offset + s * d_model + d;

                let q_spike = spikes[idx_q];
                let p_spike = spikes[idx_p];
                let n_spike = spikes[idx_n];

                let mut pull = p_spike - q_spike;
                // Suntik energi kecil jika terjadi "mati" semua
                if q_spike == 0.0 && p_spike == 0.0 && n_spike == 0.0 {
                    pull = 0.05;
                }

                let push = (q_spike * n_spike) * 0.2;

                if pull != 0.0 || push != 0.0 {
                    local_loss += pull.abs() + push;
                }
            }
        }
        local_loss
    }).collect();

    total_loss = pair_losses.iter().sum();

    // Karena err_data harus di-mutate dan par_iter mutable itu strict dengan non-overlapping slice,
    // kita gunakan pendekatan mutasi sequential atau safe chunking.
    for i in 0..num_pairs {
        let q_offset = i * sequence_length * d_model;
        let p_offset = (num_pairs + i) * sequence_length * d_model;
        let neg_idx = (i + 1) % num_pairs;
        let n_offset = (num_pairs + neg_idx) * sequence_length * d_model;

        for s in 0..sequence_length {
            for d in 0..d_model {
                let idx_q = q_offset + s * d_model + d;
                let idx_p = p_offset + s * d_model + d;
                let idx_n = n_offset + s * d_model + d;

                let q_spike = spikes[idx_q];
                let p_spike = spikes[idx_p];
                let n_spike = spikes[idx_n];

                let mut pull = p_spike - q_spike;
                if q_spike == 0.0 && p_spike == 0.0 && n_spike == 0.0 {
                    pull = 0.05;
                }
                let push = (q_spike * n_spike) * 0.2;

                if pull != 0.0 || push != 0.0 {
                    err_data[idx_q] += pull - push;
                    err_data[idx_p] += -pull;
                    err_data[idx_n] += -push;
                }
            }
        }
    }

    total_loss
}
