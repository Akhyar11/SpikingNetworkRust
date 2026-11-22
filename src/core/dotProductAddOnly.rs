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
    let dense = std::env::var("DENSE_POOLER").is_ok();

    // Eksekusi paralel per baris (batch)
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(b, mut out_row)| {
            let input_row = inputs.row(b);
            
            // E6 control: DENSE_POOLER=1 replaces the spike-gated additions by a dense multiply-accumulate
            if dense {
                for in_d in 0..input_dim {
                    let x = input_row[in_d];
                    let weight_row = weights.row(in_d);
                    for out_d in 0..output_dim { out_row[out_d] += x * weight_row[out_d]; }
                }
                return;
            }
            // Loop khusus fitur Add-Only
            for in_d in 0..input_dim {
                // Input SNN biner (0.0 atau 1.0) dari SpikingSelfAttention
                if input_row[in_d] > 0.0 {
                    let weight_row = weights.row(in_d);
                    for out_d in 0..output_dim {
                        out_row[out_d] += weight_row[out_d];
                    }
                }
            }
        });

    out
}
