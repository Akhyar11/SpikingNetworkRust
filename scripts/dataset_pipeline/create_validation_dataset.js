import { pipeline } from '@xenova/transformers';
import * as fs from 'fs';
import { join } from 'path';

const EXPERIMENT_DIR = join(import.meta.dirname, '../../experiment/file_model');

const FILES_TO_MERGE = [
    'mteb_sickr-sts.json',
    'mteb_sts12-sts.json',
    'mteb_sts13-sts.json',
    'mteb_sts14-sts.json',
    'mteb_sts15-sts.json',
    'mteb_sts16-sts.json',
    'sts-b_train.json',
    'sts-b_valid.json'
];

const OUTPUT_PATH = join(EXPERIMENT_DIR, 'all_sts_teacher_scored.json');

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
    console.log(" MERGE & SCORE DATASET VALIDASI DENGAN GURU (MiniLM-L6-v2)  ");
    console.log("============================================================");

    let combinedDataset = [];

    // 1. Baca dan Gabungkan Semua File
    for (const filename of FILES_TO_MERGE) {
        const filepath = join(EXPERIMENT_DIR, filename);
        if (fs.existsSync(filepath)) {
            const data = JSON.parse(fs.readFileSync(filepath, 'utf8'));
            let validCount = 0;
            for (const item of data) {
                // Antisipasi perbedaan nama key di file JSON
                const s1 = item.sentence1 || item.s1;
                const s2 = item.sentence2 || item.s2;
                if (s1 && s2) {
                    // Pakai format `sentence1` & `sentence2` sesuai struct Rust STSPair
                    combinedDataset.push({ sentence1: s1, sentence2: s2 });
                    validCount++;
                }
            }
            console.log(`- Dimuat ${validCount.toLocaleString()} pasang dari ${filename}`);
        } else {
            console.warn(`- Peringatan: File tidak ditemukan: ${filename}`);
        }
    }

    console.log(`\nTotal pasang kalimat untuk divalidasi: ${combinedDataset.length.toLocaleString()}`);

    // 2. Load Model Guru
    console.log("\nMemuat Model Guru (MiniLM-L6-v2) dari HuggingFace...");
    const hfExtractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

    // 3. Proses Scoring Ulang
    console.log(`\nMemulai proses scoring (Soft-Labels) untuk dataset validasi...`);
    const BATCH_SIZE = 16;
    let newDataset = [];
    const startTime = Date.now();

    for (let i = 0; i < combinedDataset.length; i += BATCH_SIZE) {
        const batch = combinedDataset.slice(i, i + BATCH_SIZE);
        
        const promises = batch.map(async (item) => {
            const outA = await hfExtractor(item.sentence1, { pooling: 'mean', normalize: true });
            const outB = await hfExtractor(item.sentence2, { pooling: 'mean', normalize: true });
            const score = standardCosine(outA.data, outB.data);
            return {
                sentence1: item.sentence1,
                sentence2: item.sentence2,
                score: Math.max(0.0, Number(score.toFixed(4)))
            };
        });

        const scoredBatch = await Promise.all(promises);
        newDataset.push(...scoredBatch);
        
        // Log & Save per 500 iterasi
        if ((i + batch.length) % 500 === 0 || (i + batch.length) === combinedDataset.length) {
            const elapsedMs = Date.now() - startTime;
            const avgSpeed = ((i + batch.length) / (elapsedMs / 1000)).toFixed(1);
            console.log(`  Progres: ${(i + batch.length).toLocaleString()} / ${combinedDataset.length.toLocaleString()} (${avgSpeed} pasang/detik)...`);
            fs.writeFileSync(OUTPUT_PATH, JSON.stringify(newDataset, null, 2));
        }
    }

    console.log(`\nSELESAI! File gabungan berhasil disimpan ke: ${OUTPUT_PATH}`);
}

main().catch(console.error);
