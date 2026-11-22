//! dump_tokens <in.json (list of strings)> <out prefix>: lowercase + BPE-encode, pad/truncate to 128.
//! Writes <out>.tok.bin (i32 little endian, [n,128], 0=pad).
use SpikingNetworkRust::core::bpe::BPETokenizer;
use std::io::Write;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let tok = BPETokenizer::load("experiment/file_model/vocab_multilingual.json");
    let s: Vec<String> = serde_json::from_reader(std::fs::File::open(&a[1]).unwrap()).unwrap();
    let mut buf = Vec::with_capacity(s.len() * 128 * 4);
    for t in &s {
        let mut ids = tok.encode(&t.to_lowercase());
        ids.truncate(128);
        for i in 0..128 { let v = if i < ids.len() { ids[i] as i32 } else { 0 }; buf.extend_from_slice(&v.to_le_bytes()); }
    }
    std::fs::File::create(format!("{}.tok.bin", a[2])).unwrap().write_all(&buf).unwrap();
    println!("{} sentences", s.len());
}
