import { pipeline } from '@xenova/transformers';
import * as fs from 'fs';

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

async function scoreDataset(inputFile, outputFile) {
    console.log(`\nMembaca dataset ${inputFile}...`);
    let data;
    try {
        data = JSON.parse(fs.readFileSync(inputFile, 'utf8'));
    } catch(e) {
        console.error("Gagal membaca file", inputFile);
        return;
    }
    
    console.log(`Dataset memiliki ${data.length} pasang kalimat.`);
    console.log("Memuat Model Guru (paraphrase-multilingual-MiniLM-L12-v2)...");
    
    const hfExtractor = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2');

    console.log("Memulai proses Teacher Scoring...");
    
    let isCheckpointLoaded = false;
    if (fs.existsSync(outputFile)) {
        try {
            const checkpointData = JSON.parse(fs.readFileSync(outputFile, 'utf8'));
            if (checkpointData.length === data.length) {
                data = checkpointData;
                isCheckpointLoaded = true;
                console.log(`Ditemukan file checkpoint. Melanjutkan proses scoring dari data sebelumnya...`);
            }
        } catch(e) {}
    }

    for (let i = 0; i < data.length; i++) {
        // Skip jika sudah ada score (berguna jika proses terhenti di tengah jalan)
        if (data[i].score !== undefined) continue;

        const item = data[i];
        try {
            // Karena output kita ada di text1 dan text2 (bukan s1 dan s2)
            const out1 = await hfExtractor(item.text1, { pooling: 'mean', normalize: true });
            const out2 = await hfExtractor(item.text2, { pooling: 'mean', normalize: true });
            const score = standardCosine(out1.data, out2.data);
            
            data[i].score = Math.max(0.0, Number(score.toFixed(4)));
        } catch (e) {
            console.error(`Gagal memproses baris ${i}:`, e.message);
            data[i].score = 0.0;
        }

        if ((i + 1) % 1000 === 0) {
            console.log(`  Progres ${inputFile}: ${(i + 1).toLocaleString()} / ${data.length.toLocaleString()} selesai...`);
            // Auto-save tiap 5000 baris
            if ((i + 1) % 5000 === 0) {
                fs.writeFileSync(outputFile, JSON.stringify(data, null, 2));
            }
        }
    }

    fs.writeFileSync(outputFile, JSON.stringify(data, null, 2));
    console.log(`SELESAI! Disimpan ke: ${outputFile}`);
}

async function main() {
    console.log("============================================================");
    console.log("  SCORING DATASET AUGMENTED MENGGUNAKAN TEACHER MODEL       ");
    console.log("============================================================");

    // Score yang eval dulu karena lebih kecil (59.000 pasang)
    await scoreDataset('../dataset_augmented_eval_1000.json', '../dataset_augmented_eval_1000_scored.json');
    
    // Setelah evaluasi selesai, lanjut ke data training (295.000 pasang)
    await scoreDataset('../dataset_augmented_5000.json', '../dataset_augmented_5000_scored.json');
}

main().catch(console.error);
