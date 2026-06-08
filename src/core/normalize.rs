use rayon::prelude::*;

/// Melakukan normalisasi L2 (Euclidean Norm) pada array 1D yang mewakili [batch_size, d_model].
/// Fungsi ini mengubah data secara in-place agar setiap vektor representasi baris (batch)
/// memiliki skala dan panjang = 1.0. Ini sangat penting untuk Stabilitas Contrastive Loss
/// dan Cosine Similarity nantinya.
pub fn l2_normalize(
    data: &mut [f32],
    batch_size: usize,
    d_model: usize
) {
    assert_eq!(data.len(), batch_size * d_model, "Dimensi data tidak cocok dengan batch_size * d_model");

    // Memecah flat array menjadi chunk berukuran `d_model` (setiap chunk mewakili 1 batch)
    // dan memprosesnya secara paralel di multi-core CPU.
    data.par_chunks_mut(d_model).for_each(|row| {
        // Hitung akar dari jumlah kuadrat (Square Root of Sum of Squares)
        let sq_sum: f32 = row.iter().map(|&x| x * x).sum();
        
        // Hindari pembagian dengan nol menggunakan epsilon kecil (1e-8)
        let norm = if sq_sum > 0.0 { sq_sum.sqrt() } else { 1e-8 };

        // Bagi setiap nilai dalam vektor baris dengan normanya
        for val in row.iter_mut() {
            *val /= norm;
        }
    });
}
