
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

    for i in 0..num_pairs {
        let q_offset = i * sequence_length * d_model;
        let p_offset = (num_pairs + i) * sequence_length * d_model;
        let neg_idx = (i + 1) % num_pairs;
        let n_offset = (num_pairs + neg_idx) * sequence_length * d_model;
        let q_len = actual_lengths[i];

        for s in 0..sequence_length {
            // MENGABAIKAN PADDING: Jangan hitung loss atau berikan energi pada token padding
            if s >= q_len {
                continue;
            }

            for d in 0..d_model {
                let idx_q = q_offset + s * d_model + d;
                let idx_p = p_offset + s * d_model + d;
                let idx_n = n_offset + s * d_model + d;

                let q_s = spikes[idx_q];
                let p_s = spikes[idx_p];
                let n_s = spikes[idx_n];

                let mut pull = p_s - q_s;
                // Suntik energi agar hidup jika kolaps total (semua mati)
                if q_s == 0.0 && p_s == 0.0 && n_s == 0.0 {
                    pull = 0.05;
                }
                
                // Daya Tolak (Repulsion): N menolak Q (hanya menolak jika keduanya spike)
                // Di sini kita pakai margin sebagai pengganti `0.2` hardcode di JS.
                let push = q_s * n_s * margin; 

                if pull != 0.0 || push != 0.0 {
                    err_data[idx_q] += pull - push; // Q ditarik ke P, didorong oleh N
                    err_data[idx_p] += -pull;       // P ditarik ke Q
                    err_data[idx_n] += -push;       // N didorong oleh Q
                    total_loss += pull.abs() + push;
                }
            }
        }
    }

    total_loss
}
