use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;

/// Operasi perkalian titik khusus untuk Spiking Neural Network (SNN).
/// Karena pada SNN input mayoritas adalah biner (1.0 atau 0.0), kita dapat menghindari
/// operasi perkalian (floating-point multiplication) yang mahal.
/// 
/// Jika input spike == 1.0, kita cukup menambahkan bobotnya (Add-Only).
/// Jika input spike == 0.0, kita abaikan.
pub fn dot_product_add_only(inputs: &ArrayView2<f32>, weights: &ArrayView2<f32>) -> Array2<f32> {
    let (batch_size, input_dim) = (inputs.nrows(), inputs.ncols());
    let output_dim = weights.ncols();
    
    assert_eq!(input_dim, weights.nrows(), "Dimensi input dan bobot tidak cocok!");

    let mut out = Array2::<f32>::zeros((batch_size, output_dim));

    // Eksekusi paralel per baris (batch)
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(b, mut out_row)| {
            let input_row = inputs.row(b);
            
            // Loop khusus fitur Add-Only
            for in_d in 0..input_dim {
                // Dalam SNN, spike biasanya > 0.5 (mendekati 1.0)
                if input_row[in_d] > 0.5 {
                    let weight_row = weights.row(in_d);
                    // Tambahkan vektor bobot secara langsung
                    for out_d in 0..output_dim {
                        out_row[out_d] += weight_row[out_d];
                    }
                }
            }
        });

    out
}
