import { pipeline } from '@xenova/transformers';
import * as fs from 'fs';
import { join } from 'path';

const EXPERIMENT_DIR = join(process.cwd(), '../experiment');
const TRAIN_INPUT_FILE = join(process.cwd(), '../dataset_20_bahasa_train.json');
const EVAL_INPUT_FILE = join(process.cwd(), '../dataset_20_bahasa_eval.json');
const TRAIN_OUTPUT_FILE = join(EXPERIMENT_DIR, 'file_model', 'teacher_distillation_dataset_20_bahasa_train.json');
const EVAL_OUTPUT_FILE = join(EXPERIMENT_DIR, 'file_model', 'teacher_distillation_dataset_20_bahasa_eval.json');

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

async function processDataset(inputFile, outputFile, maxRows = null) {
    console.log(`\nMembaca dataset dari: ${inputFile}`);
    
    if (!fs.existsSync(inputFile)) {
        console.error(`File tidak ditemukan! Pastikan Anda sudah menjalankan script download_multilingual.py terlebih dahulu.`);
        return;
    }

    const rawData = JSON.parse(fs.readFileSync(inputFile, 'utf8'));
    const dataToProcess = maxRows ? rawData.slice(0, maxRows) : rawData;
    
    console.log(`Jumlah baris yang akan di-score: ${dataToProcess.length}`);
    console.log("Memuat Model Guru (paraphrase-multilingual-MiniLM-L12-v2)...");
    
    const hfExtractor = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2');

    let finalDataset = [];
    console.log("Memulai proses Teacher Scoring...");

    for (let i = 0; i < dataToProcess.length; i++) {
        const item = dataToProcess[i];
        // Cari bahasa target (karena setiap objek berisi 'en' dan satu bahasa lain, e.g. 'id')
        const langs = Object.keys(item);
        const targetLang = langs.find(l => l !== 'en');
        
        if (!targetLang || !item.en || !item[targetLang]) continue;

        try {
            const out1 = await hfExtractor(item.en, { pooling: 'mean', normalize: true });
            const out2 = await hfExtractor(item[targetLang], { pooling: 'mean', normalize: true });
            const score = standardCosine(out1.data, out2.data);
            
            finalDataset.push({
                s1: item.en,
                s2: item[targetLang],
                score: Math.max(0.0, Number(score.toFixed(4))),
                type: `translate_en_${targetLang}`
            });
        } catch (e) {
            console.error(`Gagal memproses baris ${i}:`, e.message);
        }

        if ((i + 1) % 1000 === 0) {
            console.log(`  Progres: ${(i + 1).toLocaleString()} / ${dataToProcess.length.toLocaleString()} selesai...`);
            // Simpan progres secara berkala untuk menghindari kehilangan data
            if ((i + 1) % 5000 === 0) {
                fs.writeFileSync(outputFile, JSON.stringify(finalDataset, null, 2));
            }
        }
    }

    fs.writeFileSync(outputFile, JSON.stringify(finalDataset, null, 2));
    console.log(`SELESAI! Berhasil diberi skor dan disimpan ke: ${outputFile}`);
}

async function main() {
    console.log("============================================================");
    console.log("  SCORING DATASET 20 BAHASA UNTUK RAG DISTILLATION          ");
    console.log("============================================================");

    // Anda bisa membatasi maxRows jika ingin tes sebagian dulu, misal: maxRows = 10000
    await processDataset(TRAIN_INPUT_FILE, TRAIN_OUTPUT_FILE);
    await processDataset(EVAL_INPUT_FILE, EVAL_OUTPUT_FILE);
}

main().catch(console.error);
