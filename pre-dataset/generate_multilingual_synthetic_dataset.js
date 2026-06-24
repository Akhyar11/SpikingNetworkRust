import { pipeline } from '@xenova/transformers';
import * as fs from 'fs';
import { join } from 'path';
import * as readline from 'readline';

const EXPERIMENT_DIR = join(process.cwd(), '../experiment');
const TRAIN_OUTPUT_FILE = join(EXPERIMENT_DIR, 'file_model', 'teacher_distillation_dataset_synthetic.json');
const VALID_OUTPUT_FILE = join(EXPERIMENT_DIR, 'file_model', 'teacher_distillation_dataset_synthetic_valid.json');

// RNG yang seeded untuk reproduksibilitas
function mulberry32(a) {
    return function () {
        var t = a += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}
const seededRandom = mulberry32(42);

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

// ================= FUNGSI AUGMENTASI SINTETIS ================= //
function createWordDropout(words) {
    if (words.length < 4) return words.join(" ");
    const numDrops = seededRandom() > 0.5 ? 2 : 1;
    const w = [...words];
    for (let i = 0; i < numDrops && w.length > 2; i++) {
        const idx = Math.floor(seededRandom() * w.length);
        w.splice(idx, 1);
    }
    return w.join(" ");
}

function createSubstring(words) {
    if (words.length < 4) return words.join(" ");
    const keepRatio = 0.6 + (seededRandom() * 0.3);
    const keepCount = Math.floor(words.length * keepRatio);
    if (seededRandom() > 0.5) {
        return words.slice(0, keepCount).join(" ");
    } else {
        return words.slice(words.length - keepCount).join(" ");
    }
}

function generateSynteticSentence(sentenceA, allLines) {
    const wordsA = sentenceA.split(" ");
    let sentenceB = "";
    const randChoice = seededRandom();
    if (randChoice < 0.20) {
        sentenceB = sentenceA;
    } else if (randChoice < 0.45) {
        sentenceB = createWordDropout(wordsA);
    } else if (randChoice < 0.70) {
        sentenceB = createSubstring(wordsA);
    } else {
        const idxB = Math.floor(seededRandom() * allLines.length);
        sentenceB = allLines[idxB];
    }
    return sentenceB;
}

async function fetchTranslateBase(maxRows = 2000) {
    console.log(`Mengunduh ${maxRows} pasang base kalimat Translate (EN-ID) dari Helsinki-NLP/opus-100...`);
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
                   allRows.push(row.row.translation);
                }
            }
            offset += limit;
            if(offset % 1000 === 0) console.log(`  -> Memuat ${allRows.length} pasang base Translate...`);
        } catch (e) {
            console.error("Fetch error:", e);
            break;
        }
    }
    return allRows;
}

// Fungsi utama pembuat dataset untuk menghindari pengulangan kode
async function generateDataset(targetTotal, outputFile, hfExtractor, enLines, idLines, translateBase, splitName) {
    console.log(`\n--- Memulai proses generasi dataset sintesis: ${splitName.toUpperCase()} (Target: ${targetTotal} pasang) ---`);
    
    let dataset = [];
    if (fs.existsSync(outputFile)) {
        try {
            dataset = JSON.parse(fs.readFileSync(outputFile, 'utf8'));
            console.log(`      -> Ditemukan ${dataset.length} data lama pada ${splitName}. Kami akan meresetnya dan membuat ${targetTotal} data baru.`);
            dataset = []; 
        } catch(e) {}
    }

    for (let i = 0; i < targetTotal; i++) {
        let sentenceA = "";
        let sentenceB = "";
        let dsType = "";
        
        const randType = seededRandom();
        if (randType < 0.33) {
            dsType = "en";
            sentenceA = enLines[Math.floor(seededRandom() * enLines.length)];
            sentenceB = generateSynteticSentence(sentenceA, enLines);
        } else if (randType < 0.66) {
            dsType = "id";
            sentenceA = idLines[Math.floor(seededRandom() * idLines.length)];
            sentenceB = generateSynteticSentence(sentenceA, idLines);
        } else {
            dsType = "translate_en_id";
            const baseIdx = Math.floor(seededRandom() * translateBase.length);
            sentenceA = translateBase[baseIdx].en;
            
            const transRand = seededRandom();
            if (transRand < 0.20) {
                sentenceB = translateBase[baseIdx].id;
            } else if (transRand < 0.45) {
                sentenceB = createWordDropout(translateBase[baseIdx].id.split(" "));
            } else if (transRand < 0.70) {
                sentenceB = createSubstring(translateBase[baseIdx].id.split(" "));
            } else {
                const randB = Math.floor(seededRandom() * translateBase.length);
                sentenceB = translateBase[randB].id;
            }
        }

        const out1 = await hfExtractor(sentenceA, { pooling: 'mean', normalize: true });
        const out2 = await hfExtractor(sentenceB, { pooling: 'mean', normalize: true });
        const score = standardCosine(out1.data, out2.data);

        dataset.push({
            s1: sentenceA,
            s2: sentenceB,
            score: Math.max(0.0, Number(score.toFixed(4))),
            type: dsType
        });

        if ((i + 1) % 1000 === 0) {
            console.log(`  [${splitName}] Progres: ${(i + 1).toLocaleString()} / ${targetTotal.toLocaleString()} pasang selesai...`);
            if ((i + 1) % 5000 === 0) {
                fs.writeFileSync(outputFile, JSON.stringify(dataset, null, 2));
            }
        }
    }

    fs.writeFileSync(outputFile, JSON.stringify(dataset, null, 2));
    console.log(`\nSELESAI! ${targetTotal} dataset ${splitName} berhasil di-generate dan disimpan ke: ${outputFile}`);
}

async function main() {
    console.log("============================================================");
    console.log("      MEMBUAT DATASET DISTILASI SINTETIS (TRAIN & VALID)    ");
    console.log("============================================================");

    const corpusPath = join(EXPERIMENT_DIR, 'file_model', 'mini_corpus20mb.txt');
    const TARGET_TRAIN = 100000;
    const TARGET_VALID = 5000; // Standar jumlah validasi adalah ~5% dari data latih
    
    console.log("[1/4] Memuat Model Guru (paraphrase-multilingual-MiniLM-L12-v2)...");
    const hfExtractor = await pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2');

    console.log(`\n[2/4] Membaca corpus monolingual dari ${corpusPath}...`);
    const enLines = [];
    const idLines = [];
    
    const idWords = ['yang', 'dan', 'di', 'dari', 'ke', 'untuk', 'ini', 'itu', 'dengan', 'dalam', 'pada', 'adalah'];
    const enWords = ['the', 'and', 'in', 'of', 'to', 'for', 'this', 'that', 'with', 'on', 'is', 'are'];

    const fileStream = fs.createReadStream(corpusPath);
    const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

    for await (const line of rl) {
        const t = line.trim();
        if (t.length > 20 && t.split(" ").length <= 32) {
            const words = t.toLowerCase().split(/\s+/);
            let idCount = 0; let enCount = 0;
            for(const w of words) {
                if(idWords.includes(w)) idCount++;
                if(enWords.includes(w)) enCount++;
            }
            if(idCount > enCount) idLines.push(t);
            else enLines.push(t);
        }
    }
    console.log(`      -> Tersedia ${enLines.length} baris English, ${idLines.length} baris Indonesian.`);

    console.log("\n[3/4] Mengunduh base Translate (English vs Indonesia)...");
    const translateBase = await fetchTranslateBase(5000); 

    console.log(`\n[4/4] Memulai proses iterasi...`);
    
    // 1. Generate Dataset Training
    await generateDataset(TARGET_TRAIN, TRAIN_OUTPUT_FILE, hfExtractor, enLines, idLines, translateBase, "train");

    // 2. Generate Dataset Validasi
    await generateDataset(TARGET_VALID, VALID_OUTPUT_FILE, hfExtractor, enLines, idLines, translateBase, "validasi");

    console.log(`\n============================================================`);
    console.log(`SEMUA SELESAI! Data Train dan Validasi siap digunakan.`);
}

main().catch(console.error);
