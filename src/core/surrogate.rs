use rayon::prelude::*;

/// Mengaplikasikan Surrogate Gradient (Masking) pada sinyal error.
/// Fungsi aktivasi step (Heaviside) pada SNN tidak memiliki turunan (turunannya nol di mana-mana kecuali tak hingga di threshold).
/// Oleh karena itu kita menggunakan *Surrogate Gradient* untuk meloloskan gradien hanya jika
/// potensial membran berada di dekat threshold (di dalam *window_size*).
#[allow(non_snake_case)]
pub fn maskSurrogate(
    error_signal: &mut [f32],
    potentials: &[f32],
    threshold: &[f32],
    window_size: f32
) {
    let len = error_signal.len();
    assert_eq!(potentials.len(), len);

    error_signal.par_iter_mut()
        .zip(potentials.par_iter())
        .enumerate()
        .for_each(|(i, (err, &p))| {
            let t = if threshold.len() == 1 { threshold[0] } else { threshold[i % threshold.len()] };
            
            // Surrogate gradient berbentuk kotak / rectangular window
            if (p - t).abs() > window_size {
                *err = 0.0; // Potong gradien jika potensial terlalu jauh dari threshold
            }
        });
}
