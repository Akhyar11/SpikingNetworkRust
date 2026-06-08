use SpikingNetworkRust::layers::embedding::SpikingEmbedding;
use SpikingNetworkRust::layers::dense_bptt::SpikingDenseBPTT;

#[test]
fn test_spiking_embedding() {
    let input_dim = 100;
    let output_dim = 64;
    let mut embedding = SpikingEmbedding::new(input_dim, output_dim, 0.01, -1.0, 1.0);
    
    // Simulasi input 1 batch berukuran 3 token (Token IDs)
    // Asumsi Token ID: 5, 10, 99
    let inputs: Vec<f32> = vec![5.0, 10.0, 99.0]; 
    let spikes = embedding.forward(&inputs);
    
    // Pastikan ukuran output = batch_size * output_dim
    assert_eq!(spikes.len(), 3 * output_dim);
    
    // Pastikan semua output murni biner
    for s in &spikes {
        assert!(*s == 0.0 || *s == 1.0, "Output Embedding tidak murni binary!");
    }

    // Uji backward pass (Error signal simulasi)
    let error_signal = vec![0.1; 3 * output_dim];
    embedding.backward(&error_signal); // Harusnya sukses tanpa error bounds
}

#[test]
fn test_spiking_dense_bptt() {
    let in_features = 64;
    let units = 32;
    let batch_size = 2;
    let time_steps = 3;
    
    let mut bptt = SpikingDenseBPTT::new(in_features, units, true, -1.0, 1.0, (0.8, 0.99), (0.5, 1.0));
    
    // Wajib reset di awal kalimat
    bptt.reset_sequence(batch_size, time_steps);
    
    // ============================================
    // Forward Pass (Loop melalui 3 Time Steps)
    // ============================================
    let inputs_t0 = vec![1.0; batch_size * in_features];
    let spikes_t0 = bptt.compute_step(&inputs_t0, 0);
    assert_eq!(spikes_t0.len(), batch_size * units);
    for s in &spikes_t0 { assert!(*s == 0.0 || *s == 1.0); }
    
    let inputs_t1 = vec![0.0; batch_size * in_features];
    let spikes_t1 = bptt.compute_step(&inputs_t1, 1);
    
    let inputs_t2 = vec![1.0; batch_size * in_features];
    let spikes_t2 = bptt.compute_step(&inputs_t2, 2);

    // ============================================
    // Backward Pass / BPTT
    // ============================================
    let error_t0 = vec![0.1; batch_size * units];
    let error_t1 = vec![-0.1; batch_size * units];
    let error_t2 = vec![0.05; batch_size * units];
    let error_sequence = vec![error_t0, error_t1, error_t2];
    
    bptt.learn_through_time(&error_sequence, 0.01); 
    // Jika tak ada panic / array index out of bounds, tes ini lulus.
}

#[test]
fn test_spiking_self_attention() {
    use SpikingNetworkRust::layers::self_attention::SpikingSelfAttention;
    use SpikingNetworkRust::layers::base::Layer;

    let d_model = 16;
    let seq_length = 4;
    let batch_size = 2;
    
    let mut attention = SpikingSelfAttention::new(d_model, seq_length, 0.01, -1.0, 1.0, (0.8, 0.99), (0.1, 0.3));

    // Dummy inputs: batch_size * seq_length * d_model
    let mut inputs = vec![0.0; batch_size * seq_length * d_model];
    // Fill some random spikes
    inputs[0] = 1.0;
    inputs[5] = 1.0;
    inputs[16] = 1.0;
    inputs[25] = 1.0;

    let output = attention.forward(&inputs);
    assert_eq!(output.len(), batch_size * seq_length * d_model);

    // Test learning
    let mut error_signal = vec![0.0; batch_size * seq_length * d_model];
    error_signal[0] = 0.5;
    attention.learn_attention(&error_signal);

    // Test summary
    attention.summary();
}
