use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

const PAD_TOKEN: &str = "<PAD>";
const UNK_TOKEN: &str = "<UNK>";
const BOS_TOKEN: &str = "<BOS>";
const EOS_TOKEN: &str = "<EOS>";
const WORD_BOUNDARY: &str = "▁";
const PAIR_SEPARATOR: &str = "\0";

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BPEConfigSerde {
    pub vocab_size: u32,
    pub min_frequency: u32,
    pub pre_tokenizer: String,
    #[serde(default)]
    pub special_tokens: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct BPEVocabData {
    pub vocab: HashMap<String, u32>,
    pub merges: Vec<Vec<String>>,
    pub config: BPEConfigSerde,
}

pub struct BPETokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    merges: Vec<[String; 2]>,
    merge_ranks: HashMap<String, usize>,
    config: BPEConfigSerde,
}

impl BPETokenizer {
    pub fn load(filepath: &str) -> Self {
        let file = File::open(filepath).expect("Failed to open vocab file");
        let reader = BufReader::new(file);
        let data: BPEVocabData = serde_json::from_reader(reader).expect("Failed to parse JSON");

        let mut reverse_vocab = HashMap::new();
        for (token, &id) in &data.vocab {
            reverse_vocab.insert(id, token.clone());
        }

        let mut merges = Vec::new();
        let mut merge_ranks = HashMap::new();
        for (i, m) in data.merges.into_iter().enumerate() {
            if m.len() == 2 {
                let left = m[0].clone();
                let right = m[1].clone();
                let pair_key = format!("{}{}{}", left, PAIR_SEPARATOR, right);
                merges.push([left, right]);
                merge_ranks.insert(pair_key, i);
            }
        }

        Self {
            vocab: data.vocab,
            reverse_vocab,
            merges,
            merge_ranks,
            config: data.config,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let words = self.pre_tokenize(text);
        let mut token_ids = Vec::new();

        for word in words {
            if word.is_empty() {
                continue;
            }
            let word_token_ids = self.encode_word(&word);
            token_ids.extend(word_token_ids);
        }

        token_ids
    }

    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let bos = *self.vocab.get(BOS_TOKEN).unwrap_or(&0);
        let eos = *self.vocab.get(EOS_TOKEN).unwrap_or(&0);
        
        let mut ids = vec![bos];
        ids.extend(self.encode(text));
        ids.push(eos);
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut tokens = Vec::new();
        for &id in ids {
            if let Some(token) = self.reverse_vocab.get(&id)
                && token != BOS_TOKEN && token != EOS_TOKEN && token != PAD_TOKEN {
                    tokens.push(token.clone());
                }
        }
        let joined = tokens.join("");
        joined.replace(WORD_BOUNDARY, " ").trim().to_string()
    }

    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            if c.is_whitespace() {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            } else {
                current.push(c);
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    fn encode_word(&self, word: &str) -> Vec<u32> {
        let full_word = format!("{}{}", WORD_BOUNDARY, word);

        // Optional: lookup whole word first
        if let Some(&id) = self.vocab.get(&full_word) {
            return vec![id];
        }

        let mut symbols = self.create_initial_symbols(&full_word);
        self.apply_merge_rules_in_place(&mut symbols);

        let mut word_token_ids = Vec::new();
        for sym in symbols {
            if let Some(&id) = self.vocab.get(&sym) {
                word_token_ids.push(id);
            } else if let Some(&unk) = self.vocab.get(UNK_TOKEN) {
                word_token_ids.push(unk);
            }
        }

        word_token_ids
    }

    fn create_initial_symbols(&self, token: &str) -> Vec<String> {
        if let Some(body) = token.strip_prefix(WORD_BOUNDARY) {
            let mut syms = vec![WORD_BOUNDARY.to_string()];
            for c in body.chars() {
                syms.push(c.to_string());
            }
            syms
        } else {
            token.chars().map(|c| c.to_string()).collect()
        }
    }

    fn apply_merge_rules_in_place(&self, symbols: &mut Vec<String>) {
        while symbols.len() > 1 {
            let mut best_rank = usize::MAX;
            let mut best_index = None;

            for i in 0..symbols.len() - 1 {
                let pair_key = format!("{}{}{}", symbols[i], PAIR_SEPARATOR, symbols[i + 1]);
                if let Some(&rank) = self.merge_ranks.get(&pair_key)
                    && rank < best_rank {
                        best_rank = rank;
                        best_index = Some(i);
                    }
            }

            if let Some(i) = best_index {
                let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                symbols[i] = merged;
                symbols.remove(i + 1);
            } else {
                break;
            }
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn pad_id(&self) -> u32 {
        *self.vocab.get(PAD_TOKEN).unwrap_or(&0)
    }

    pub fn bos_id(&self) -> u32 {
        *self.vocab.get(BOS_TOKEN).unwrap_or(&0)
    }
}
