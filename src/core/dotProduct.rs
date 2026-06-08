use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;

/// Operasi perkalian titik penuh (Full Dot Product) untuk continuous/analog inputs.
pub fn dot_product(inputs: &ArrayView2<f32>, weights: &ArrayView2<f32>) -> Array2<f32> {
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
            
            // Lakukan perkalian titik (dot product) penuh untuk mendukung continuous input (analog current)
            for in_d in 0..input_dim {
                let in_val = input_row[in_d];
                // Abaikan nol persis untuk optimasi sparsity
                if in_val != 0.0 {
                    let weight_row = weights.row(in_d);
                    for out_d in 0..output_dim {
                        out_row[out_d] += in_val * weight_row[out_d];
                    }
                }
            }
        });

    out
}
