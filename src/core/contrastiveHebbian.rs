
/// Kalkulasi Fungsi Loss "In-Batch Negative Contrastive" khusus SNN.
/// Menarik vektor Q ke arah Positif (P) dan menjauhkannya dari Negatif (N).
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
    
    // Cek apakah batch mengandung Hard Negative (3 * num_pairs)
    let total_samples = actual_lengths.len();
    let has_hard_negative = total_samples >= 3 * num_pairs;

    for i in 0..num_pairs {
        let q_offset = i * sequence_length * d_model;
        let p_offset = (num_pairs + i) * sequence_length * d_model;
        
        // In-batch Negative (menggunakan P dari pasangan lain)
        let neg_idx = (i + 1) % num_pairs;
        let n1_offset = (num_pairs + neg_idx) * sequence_length * d_model;
        
        // Hard Negative (jika ada, berada di indeks 2 * num_pairs + i)
        let n2_offset = if has_hard_negative {
            (2 * num_pairs + i) * sequence_length * d_model
        } else {
            0
        };
        
        let q_len = actual_lengths[i];

        for s in 0..sequence_length {
            // MENGABAIKAN PADDING
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

                let mut pull = p_s - q_s;
                if q_s == 0.0 && p_s == 0.0 && n1_s == 0.0 {
                    pull = 0.05;
                }
                
                // Daya Tolak (Repulsion):
                // 1. Dari In-batch negative
                let push1 = q_s * n1_s * margin; 
                // 2. Dari Hard negative (bobot penolakan dikurangi menjadi 0.2 sesuai permintaan)
                let push2 = if has_hard_negative { q_s * n2_s * margin * 0.2 } else { 0.0 };
                
                let total_push = push1 + push2;

                if pull != 0.0 || total_push != 0.0 {
                    err_data[idx_q] += pull - total_push; // Q ditarik ke P, didorong oleh N1 & N2
                    err_data[idx_p] += -pull;             // P ditarik ke Q
                    err_data[idx_n1] += -push1;           // N1 didorong oleh Q
                    if has_hard_negative {
                        let idx_n2 = n2_offset + s * d_model + d;
                        err_data[idx_n2] += -push2;       // N2 (Hard Negative) didorong lebih kuat
                    }
                    total_loss += pull.abs() + total_push;
                }
            }
        }
    }

    total_loss
}

/// Kalkulasi Fungsi Loss Distilasi Hebbian.
/// Menggunakan skor target dari model Guru sebagai bobot Tarikan (Pull) dan Tolakan (Push)
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
        
        // Target dari Guru (MiniLM)
        let target_score = target_scores[i].clamp(0.0, 1.0);
        let pull_weight = target_score;       // Makin mirip (mendekati 1), makin ditarik
        let push_weight = 1.0 - target_score; // Makin beda (mendekati 0), makin didorong
        
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

                // PULL: Tarik vektor A menuju B
                let mut pull = b_s - a_s;
                if a_s == 0.0 && b_s == 0.0 {
                    pull = 0.05; // Stimulus firing basal
                }
                pull *= pull_weight;

                // PUSH: Dorong vektor A menjauh dari B jika mereka spike secara bersamaan
                let push = a_s * b_s * margin * push_weight;
                
                if pull != 0.0 || push != 0.0 {
                    // Update secara simetris
                    err_data[idx_a] += pull - push; 
                    err_data[idx_b] += -pull - push; // B ditarik ke A (-pull) dan saling tolak (-push)
                    
                    total_loss += pull.abs() + push;
                }
            }
        }
    }

    total_loss
}
