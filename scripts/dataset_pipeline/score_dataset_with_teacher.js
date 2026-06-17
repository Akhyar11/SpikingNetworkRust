import { pipeline } from '@xenova/transformers';
import * as fs from 'fs';
import { join } from 'path';

// Kita menggunakan path dari dataset yang baru dibuat (250k data)
const EXPERIMENT_DIR = join(import.meta.dirname, '../../experiment/file_model');
const INPUT_PATH = join(EXPERIMENT_DIR, 'teacher_distillation_dataset.json');
// Kita simpan ke file dengan nama sedikit berbeda agar data asli tidak hilang
const OUTPUT_PATH = join(EXPERIMENT_DIR, 'teacher_distillation_dataset_scored.json');

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

async function main() {
    console.log("============================================================");
    console.log(" MENGHITUNG SOFT-LABELS (DISTILASI) DARI GURU (MiniLM-L6-v2)  ");
    console.log("============================================================");

    console.log("[1/3] Membaca dataset baru (SNLI)...");
    if (!fs.existsSync(INPUT_PATH)) {
        console.error("File dataset tidak ditemukan!");
        return;
    }
    
    let dataset = JSON.parse(fs.readFileSync(INPUT_PATH, 'utf8'));
    console.log(`      -> Ditemukan ${dataset.length} pasang kalimat.`);

    console.log("[2/3] Memuat Model Guru (MiniLM-L6-v2) dari HuggingFace...");
    const hfExtractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

    console.log(`[3/3] Memulai proses distilasi ulang...`);
    console.log(`      (PERINGATAN: Memproses ratusan ribu data dengan CPU Node.js mungkin membutuhkan waktu 1-2 jam)`);
    
    const BATCH_SIZE = 16;
    let newDataset = [];
    
    // Resume (Jika ada file output yang sudah setengah jadi)
    if (fs.existsSync(OUTPUT_PATH)) {
        try {
            newDataset = JSON.parse(fs.readFileSync(OUTPUT_PATH, 'utf8'));
            console.log(`      -> MELANJUTKAN dari baris ke-${newDataset.length}...`);
        } catch (e) {
            newDataset = [];
        }
    }

    const startIndex = newDataset.length;
    const startTime = Date.now();

    for (let i = startIndex; i < dataset.length; i += BATCH_SIZE) {
        const batch = dataset.slice(i, i + BATCH_SIZE);
        
        // Memproses satu batch sekaligus dengan Promise.all untuk mempercepat
        const promises = batch.map(async (item) => {
            const outA = await hfExtractor(item.s1, { pooling: 'mean', normalize: true });
            const outB = await hfExtractor(item.s2, { pooling: 'mean', normalize: true });
            const score = standardCosine(outA.data, outB.data);
            return {
                s1: item.s1,
                s2: item.s2,
                score: Math.max(0.0, Number(score.toFixed(4))) // Pastikan tidak ada skor negatif
            };
        });

        const scoredBatch = await Promise.all(promises);
        newDataset.push(...scoredBatch);
        
        // Cetak progres tiap kelipatan 500
        if ((i + batch.length) % 500 === 0 || (i + batch.length) === dataset.length) {
            const elapsedMs = Date.now() - startTime;
            const doneThisRun = (i + batch.length) - startIndex;
            const avgSpeed = (doneThisRun / (elapsedMs / 1000)).toFixed(1);
            
            console.log(`  Progres Distilasi: ${(i + batch.length).toLocaleString()} / ${dataset.length.toLocaleString()} (${avgSpeed} pasang/detik)...`);
            
            // Auto save agar aman jika dicancel / terputus
            fs.writeFileSync(OUTPUT_PATH, JSON.stringify(newDataset, null, 2));
        }
    }

    console.log(`\nSELESAI! Dataset telah memiliki "Soft-Labels" murni dari pemikiran Model Guru.`);
    console.log(`Tersimpan ke: ${OUTPUT_PATH}`);
    console.log(`\nSilakan ganti nilai dataset_path di src/bin/train_distil_only.rs menjadi:`);
    console.log(`let dataset_path = "experiment/file_model/teacher_distillation_dataset_scored.json";`);
}

main().catch(console.error);
