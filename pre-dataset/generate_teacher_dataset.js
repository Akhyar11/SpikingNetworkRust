import { pipeline } from '@xenova/transformers';
import * as fs from 'fs';
import * as readline from 'readline';
import { join } from 'path';

// Seeded RNG for reproducibility
function mulberry32(a) {
    return function () {
        var t = a += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}
const seededRandom = mulberry32(42); // fixed seed 42

const EXPERIMENT_DIR = join(import.meta.dirname, '../experiment');

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

// Fungsi untuk Word Dropout (Menghapus 1-2 kata secara acak agar terkesan seperti typo/hilang konteks)
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

// Fungsi untuk Substring (Mengambil sebagian kalimat, meniru input pengguna yang tidak lengkap)
function createSubstring(words) {
    if (words.length < 4) return words.join(" ");
    const keepRatio = 0.6 + (seededRandom() * 0.3); // Ambil 60% - 90% kata
    const keepCount = Math.floor(words.length * keepRatio);
    if (seededRandom() > 0.5) {
        return words.slice(0, keepCount).join(" "); // Potong dari depan
    } else {
        return words.slice(words.length - keepCount).join(" "); // Potong dari belakang
    }
}

// Fungsi untuk memuat dataset anotasi manusia (STS-B English & Indonesian)
function loadHumanDatasets() {
    let combined = [];

    // 1. Load sts-b_train.json
    try {
        const stsJson = JSON.parse(fs.readFileSync(join(EXPERIMENT_DIR, 'sts-b_train.json'), 'utf8'));
        let count = 0;
        stsJson.forEach(item => {
            if (item.sentence1 && item.sentence2 && item.score !== undefined) {
                combined.push({
                    s1: item.sentence1,
                    s2: item.sentence2,
                    score: Number(item.score)
                });
                count++;
            }
        });
        console.log(`      -> Berhasil memuat ${count} pasang dari sts-b_train.json`);
    } catch (err) {
        console.error("      -> Gagal memuat sts-b_train.json:", err.message);
    }

    // 2. Load data_stsb.train.modified_indo.csv (Manual CSV Parser agar tahan banting dengan quotes)
    try {
        const csvString = fs.readFileSync(join(EXPERIMENT_DIR, 'data_stsb.train.modified_indo.csv'), 'utf8');
        let rows = [];
        let current = [];
        let cell = '';
        let inQuotes = false;
        for (let i = 0; i < csvString.length; i++) {
            const char = csvString[i];
            if (inQuotes) {
                if (char === '"') {
                    if (csvString[i + 1] === '"') { cell += '"'; i++; }
                    else { inQuotes = false; }
                } else { cell += char; }
            } else {
                if (char === '"') { inQuotes = true; }
                else if (char === ',') { current.push(cell); cell = ''; }
                else if (char === '\n' || char === '\r') {
                    if (char === '\r' && csvString[i + 1] === '\n') i++;
                    current.push(cell); rows.push(current);
                    current = []; cell = '';
                } else { cell += char; }
            }
        }
        if (cell || current.length) { current.push(cell); rows.push(current); }

        let validRows = 0;
        // Skip header (i=1)
        for (let i = 1; i < rows.length; i++) {
            if (rows[i].length >= 3) {
                const s1 = rows[i][0].trim();
                const s2 = rows[i][1].trim();
                const scoreStr = rows[i][rows[i].length - 1].trim(); // Ambil kolom terakhir
                const score = parseFloat(scoreStr);
                if (s1 && s2 && !isNaN(score)) {
                    combined.push({ s1: s1, s2: s2, score: score });
                    validRows++;
                }
            }
        }
        console.log(`      -> Berhasil memuat ${validRows} pasang dari data_stsb.train.modified_indo.csv`);
    } catch (err) {
        console.error("      -> Gagal memuat CSV Indo:", err.message);
    }

    return combined;
}

async function main() {
    console.log("============================================================");
    console.log("      MEMBUAT DATASET DISTILASI (TEACHER: MiniLM-L6-v2)     ");
    console.log("============================================================");

    const corpusPath = join(EXPERIMENT_DIR, 'mini_corpus20mb.txt'); // Wikipedia Indo & Inggris
    const outputPath = join(EXPERIMENT_DIR, 'teacher_distillation_dataset.json');
    const numPairsToGenerate = 100000; // Target 100 Ribu Pasang Kalimat (Anda bisa ubah ini)

    console.log("[1/3] Memuat Model Guru (MiniLM-L6-v2) dari HuggingFace...");
    const hfExtractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

    console.log(`[2/3] Membaca corpus dari ${corpusPath}...`);
    const allLines = [];
    const fileStream = fs.createReadStream(corpusPath);
    const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

    for await (const line of rl) {
        const t = line.trim();
        // Syarat kalimat yang baik untuk distilasi SNN:
        // 1. Tidak terlalu pendek (> 20 karakter)
        // 2. Tidak terlalu panjang (<= 32 kata, menyesuaikan Max_Seq SNN kita)
        if (t.length > 20) {
            const wordCount = t.split(" ").length;
            if (wordCount <= 32) {
                // Gunakan teknik Reservoir Sampling sederhana:
                // Masukkan semua kalimat yang valid, nanti kita acak (shuffle) agar 
                // proporsi Bahasa Indonesia dan Inggris bercampur rata.
                allLines.push(t);
            }
        }
    }

    // Mengacak (Shuffle) seluruh corpus agar bahasa tidak mengumpul di awal/akhir
    for (let i = allLines.length - 1; i > 0; i--) {
        const j = Math.floor(seededRandom() * (i + 1));
        [allLines[i], allLines[j]] = [allLines[j], allLines[i]];
    }

    // Jika terlalu besar, baru potong di sini (setelah diacak, sehingga bahasa proporsional)
    if (allLines.length > 300000) {
        allLines.length = 300000;
    }

    console.log(`      -> Berhasil memuat dan mengacak ${allLines.length} kalimat valid (Maks 32 Kata).`);

    console.log("\n[3/4] Menggabungkan Dataset STS-B Manusia (Inggris & Indo)...");
    const dataset = loadHumanDatasets();
    console.log(`      -> Total pasang kalimat manusia: ${dataset.length.toLocaleString()}`);

    console.log(`\n[4/4] Memulai proses distilasi tambahan (Target: +${numPairsToGenerate} pasang dari MiniLM)...`);

    // Simpan progres ke file setiap 1000 iterasi agar aman jika terputus
    for (let i = 0; i < numPairsToGenerate; i++) {
        // Ambil kalimat asli secara acak
        const idxA = Math.floor(seededRandom() * allLines.length);
        const sentenceA = allLines[idxA];
        const wordsA = sentenceA.split(" ");

        let sentenceB = "";

        // Strategi Sampling untuk Variasi Dataset (LEBIH MASUK AKAL):
        const randChoice = seededRandom();
        if (randChoice < 0.20) {
            // 20%: Identik / Hampir Identik (Positif Kuat)
            sentenceB = sentenceA;
        } else if (randChoice < 0.45) {
            // 25%: Word Dropout (Kalimat dengan kata yang hilang)
            sentenceB = createWordDropout(wordsA);
        } else if (randChoice < 0.70) {
            // 25%: Substring / Truncation (Kalimat yang terpotong)
            sentenceB = createSubstring(wordsA);
        } else {
            // 30%: Pure Random (Negatif Murni)
            const idxB = Math.floor(seededRandom() * allLines.length);
            sentenceB = allLines[idxB];
        }

        // Tanyakan ke Guru: "Berapa kemiripannya?"
        const hfOutA = await hfExtractor(sentenceA, { pooling: 'mean', normalize: true });
        const hfOutB = await hfExtractor(sentenceB, { pooling: 'mean', normalize: true });
        const teacherScore = standardCosine(hfOutA.data, hfOutB.data);

        // Simpan ke dataset (hanya skor positif antara 0.0 - 1.0)
        dataset.push({
            s1: sentenceA,
            s2: sentenceB,
            score: Math.max(0.0, Number(teacherScore.toFixed(4))) // Cegah nilai minus
        });

        // Print progress
        if ((i + 1) % 1000 === 0) {
            console.log(`  Progres Distilasi: ${(i + 1).toLocaleString()} / ${numPairsToGenerate.toLocaleString()} pasang selesai...`);
            // Auto-save tiap 5000 step
            if ((i + 1) % 5000 === 0) {
                fs.writeFileSync(outputPath, JSON.stringify(dataset, null, 2));
            }
        }
    }

    // Save Final
    fs.writeFileSync(outputPath, JSON.stringify(dataset, null, 2));
    console.log(`\nSELESAI! Dataset berhasil di-generate dan disimpan ke: ${outputPath}`);
}

main().catch(console.error);
