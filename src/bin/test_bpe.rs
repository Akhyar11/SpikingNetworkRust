use SpikingNetworkRust::core::bpe::BPETokenizer;

fn main() {
    let vocab_path = "/home/akhyar/Dokumen/Code/NODE_JS/penelitian_model_bahasa_dengan_spiking/models/vocab.json";
    println!("Mencoba memuat vocab dari: {}", vocab_path);
    
    let tokenizer = BPETokenizer::load(vocab_path);
    println!("✅ Vocab berhasil dimuat!");
    println!("Ukuran Vocab: {}", tokenizer.vocab_size());
    println!("PAD ID: {}", tokenizer.pad_id());
    println!("BOS ID: {}", tokenizer.bos_id());

    let text = "Halo, apakah BPE Rust ini bisa memuat vocab JS dengan baik?";
    println!("\nTeks asli: {}", text);
    
    let encoded = tokenizer.encode(text);
    println!("Hasil Encode (Token IDs): {:?}", encoded);
    
    let decoded = tokenizer.decode(&encoded);
    println!("Hasil Decode: {}", decoded);
}
