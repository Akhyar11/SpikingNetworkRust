import { pipeline } from '@xenova/transformers';
import * as fs from 'fs';
import { join } from 'path';

const EXPERIMENT_DIR = join(process.cwd(), '../experiment');
const TRAIN_OUTPUT_FILE = join(EXPERIMENT_DIR, 'file_model', 'teacher_distillation_dataset_scored.json');

function standardCosine(vecA, vecB) {
    let dot = 0.0, normA = 0.0, normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
        dot += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// 1. Fetch Translate (English-Indonesian) dari Opus-100 (Translasi Asli)
async function fetchTranslateOpus(maxRows = 20000) {
    console.log(`Mengunduh ${maxRows} data Translate (EN-ID) dari Helsinki-NLP/opus-100...`);
    let allRows = [];
    let offset = 0;
    const limit = 100;
    
    while (offset < maxRows) {
        const url = `https://datasets-server.huggingface.co/rows?dataset=Helsinki-NLP/opus-100&config=en-id&split=train&offset=${offset}&length=${limit}`;
        try {
            const res = await fetch(url);
            if (!res.ok) break;
            const data = await res.json();
            if (!data.rows || data.rows.length === 0) break;
            
            for (const row of data.rows) {
                if(row.row && row.row.translation) {
                   allRows.push({
                       s1: row.row.translation.en,
                       s2: row.row.translation.id,
                       type: "translate_en_id"
                   });
                }
            }
            offset += limit;
            if(offset % 2000 === 0) console.log(`  -> Memuat ${allRows.length} baris Translate...`);
        } catch (e) {
            console.error("Fetch error:", e);
            break;
        }
    }
    return allRows;
}

// 2. Fetch English (EN-EN) dari SNLI (Stanford Natural Language Inference)
async function fetchEnglishSNLI(maxRows = 20000) {
    console.log(`Mengunduh ${maxRows} data English (EN-EN) dari SNLI...`);
    let allRows = [];
    let offset = 0;
    const limit = 100;
    
    while (offset < maxRows) {
        const url = `https://datasets-server.huggingface.co/rows?dataset=snli&config=plain_text&split=train&offset=${offset}&length=${limit}`;
        try {
            const res = await fetch(url);
            if (!res.ok) break;
            const data = await res.json();
            if (!data.rows || data.rows.length === 0) break;
            
            for (const row of data.rows) {
                if(row.row && row.row.premise && row.row.hypothesis) {
                   allRows.push({
                       s1: row.row.premise,
                       s2: row.row.hypothesis,
                       type: "en"
                   });
                }
            }
            offset += limit;
            if(offset % 2000 === 0) console.log(`  -> Memuat ${allRows.length} baris English...`);
        } catch (e) {
            console.error("Fetch error:", e);
            break;
        }
    }
    return allRows;
}

// 3. Fetch Indonesian (ID-ID) dari IndoNLI (10.000 data) + Lokal STSB Indo (12.000 data) -> Total > 22.000 data
async function fetchIndonesianDatasets() {
    console.log(`Mengunduh data Indonesian (ID-ID) dari IndoNLI Github...`);
    let allRows = [];
    
    // a. Fetch dari IndoNLI
    try {
        const res = await fetch("https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/train.jsonl");
        const text = await res.text();
        const lines = text.split("\n");
        for (const line of lines) {
            if (!line.trim()) continue;
            try {
                const obj = JSON.parse(line);
                if (obj.premise && obj.hypothesis) {
                    allRows.push({
                        s1: obj.premise,
                        s2: obj.hypothesis,
                        type: "id"
                    });
                }
            } catch(e) {}
        }
        console.log(`  -> Berhasil memuat ${allRows.length} baris dari IndoNLI.`);
    } catch (e) {
        console.error("Gagal mendownload IndoNLI:", e);
    }

    // b. Tambahkan dari data_stsb.train.modified_indo.csv lokal (12.131 data)
    const localCsvPath = join(EXPERIMENT_DIR, 'file_model', 'data_stsb.train.modified_indo.csv');
    try {
        if (fs.existsSync(localCsvPath)) {
            const csvString = fs.readFileSync(localCsvPath, 'utf8');
            const lines = csvString.split('\n');
            let valid = 0;
            // skip header
            for (let i = 1; i < lines.length; i++) {
                const parts = lines[i].split(',');
                if (parts.length >= 3) {
                    const s1 = parts[0].trim();
                    // karena CSV kadang ada koma di tengah kalimat, kita ambil s1 dan s2 yang cukup aman
                    // Untuk kesederhanaan, ambil s1 dan kolom sebelum score
                    const s2 = parts[parts.length - 2].trim();
                    if (s1 && s2) {
                        allRows.push({ s1, s2, type: "id" });
                        valid++;
                    }
                }
            }
            console.log(`  -> Berhasil memuat tambahan ${valid} baris dari CSV Lokal Indo.`);
        }
    } catch (e) {
        console.error("Gagal membaca CSV lokal:", e);
    }
    
    return allRows; // Pasti lebih dari 20.000
}

async function main() {
    console.log("============================================================");
    console.log("  MENGUNDUH 60.000+ REAL DATASET MULTILINGUAL & SCORING     ");
    console.log("============================================================");

    // 1. Ambil Data Real dari Internet
    const translatePairs = await fetchTranslateOpus(20000);   // ~20.000 data
    const englishPairs = await fetchEnglishSNLI(20000);       // ~20.000 data
    const indonesianPairs = await fetchIndonesianDatasets();  // ~22.000+ data

    const allData = [...translatePairs, ...englishPairs, ...indonesianPairs];
    console.log(`\nTOTAL REAL DATA YANG SIAP DI-SCORE: ${allData.length} baris!`);

    console.log("\n[2/2] Memuat Model Guru (paraphrase-multilingual-MiniLM-L12-v2)...");
    const hfExtractor = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2');

    console.log(`\nMemulai proses Teacher Scoring (Proses inferensi akan memakan waktu)...`);
    
    let finalDataset = [];
    if (fs.existsSync(TRAIN_OUTPUT_FILE)) {
        try {
            // Jika mau menggabung data sebelumnya, uncomment ini
            // finalDataset = JSON.parse(fs.readFileSync(TRAIN_OUTPUT_FILE, 'utf8'));
            // console.log(`      -> Ditemukan data lama. Mulai dataset baru agar bersih 60.000.`);
        } catch(e) {}
    }

    // Melakukan Scoring untuk semua 60.000+ data
    for (let i = 0; i < allData.length; i++) {
        const item = allData[i];
        
        try {
            const out1 = await hfExtractor(item.s1, { pooling: 'mean', normalize: true });
            const out2 = await hfExtractor(item.s2, { pooling: 'mean', normalize: true });
            const score = standardCosine(out1.data, out2.data);
            
            finalDataset.push({
                s1: item.s1,
                s2: item.s2,
                score: Math.max(0.0, Number(score.toFixed(4))),
                type: item.type
            });
        } catch (e) {
            console.error(`Gagal memproses baris ${i}:`, e.message);
        }

        if ((i + 1) % 1000 === 0) {
            console.log(`  Progres Scoring: ${(i + 1).toLocaleString()} / ${allData.length.toLocaleString()} pasang selesai...`);
            if ((i + 1) % 5000 === 0) {
                fs.writeFileSync(TRAIN_OUTPUT_FILE, JSON.stringify(finalDataset, null, 2));
            }
        }
    }

    fs.writeFileSync(TRAIN_OUTPUT_FILE, JSON.stringify(finalDataset, null, 2));
    console.log(`\nSELESAI! ${finalDataset.length} data asli dari internet berhasil diberi skor dan disimpan ke: ${TRAIN_OUTPUT_FILE}`);
}

main().catch(console.error);
