use SpikingNetworkRust::core::bpe::BPETokenizer;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let text = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "こんにちは世界".to_string()
    };

    let tokenizer = BPETokenizer::load("experiment/file_model/vocab_multilingual.json");
    println!("Vocab size: {}", tokenizer.vocab_size());
    println!("Original text: {}", text);

    let tokens = tokenizer.encode(&text);
    println!("Encoded tokens ({} tokens): {:?}", tokens.len(), tokens);

    let decoded = tokenizer.decode(&tokens);
    println!("Decoded text: {}", decoded);
    println!("Match: {}", text == decoded);
}
