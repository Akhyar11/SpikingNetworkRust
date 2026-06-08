use rayon::prelude::*;

/// Operasi Leaky Integrate-and-Fire (LIF) Step secara Paralel.
/// 
/// Menyelesaikan persamaan:
/// V[t] = (V[t-1] * beta) + dot_product_input
/// spikes = V[t] > threshold ? 1.0 : 0.0
/// V[t] = spikes > 0 ? 0.0 : V[t]  (Soft/Hard Reset)
#[allow(non_snake_case)]
pub fn lifStep(
    potentials: &mut [f32],
    dot: &[f32],
    spikes: &mut [f32],
    lastPotentials: &mut [f32],
    beta: &[f32],
    threshold: &[f32]
) {
    let len = potentials.len();
    assert_eq!(dot.len(), len);
    assert_eq!(spikes.len(), len);
    assert_eq!(lastPotentials.len(), len);

    potentials.par_iter_mut()
        .zip(dot.par_iter())
        .zip(spikes.par_iter_mut())
        .zip(lastPotentials.par_iter_mut())
        .enumerate()
        .for_each(|(i, (((p, &d), s), lp))| {
            let b = if beta.len() == 1 { beta[0] } else { beta[i % beta.len()] };
            let t = if threshold.len() == 1 { threshold[0] } else { threshold[i % threshold.len()] };

            let current_p = f32::min(1.0, (*p * b) + d);
            
            // Simpan potensi sesaat (sebelum direset) untuk keperluan Surrogate Gradient
            *lp = current_p;

            if current_p >= t {
                *s = 1.0;
                *p = current_p - t; // Soft reset (mencegah information loss)
            } else {
                *s = 0.0;
                *p = current_p;
            }
        });
}
