import * as fs from 'fs';
import { join } from 'path';

// Seeded RNG for reproducibility
function mulberry32(a) {
    return function() {
      var t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
}
const seededRandom = mulberry32(42); // fixed seed 42

const EXPERIMENT_DIR = join(import.meta.dirname, '../../experiment/file_model');

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
        console.log(`[+] Berhasil memuat ${count} pasang dari sts-b_train.json`);
    } catch (err) {
        console.error("[-] Gagal memuat sts-b_train.json:", err.message);
    }

    // 2. Load data_stsb.train.modified_indo.csv
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
                    if (char === '\r' && csvString[i+1] === '\n') i++;
                    current.push(cell); rows.push(current);
                    current = []; cell = '';
                } else { cell += char; }
            }
        }
        if (cell || current.length) { current.push(cell); rows.push(current); }

        let validRows = 0;
        for (let i = 1; i < rows.length; i++) {
            if (rows[i].length >= 3) {
                const s1 = rows[i][0].trim();
                const s2 = rows[i][1].trim();
                const scoreStr = rows[i][rows[i].length - 1].trim();
                const score = parseFloat(scoreStr);
                if (s1 && s2 && !isNaN(score)) {
                    combined.push({ s1: s1, s2: s2, score: score });
                    validRows++;
                }
            }
        }
        console.log(`[+] Berhasil memuat ${validRows} pasang dari data_stsb.train.modified_indo.csv`);
    } catch (err) {
        console.error("[-] Gagal memuat CSV Indo:", err.message);
    }

    return combined;
}

function main() {
    console.log("============================================================");
    console.log("      MEMBUAT DATASET KHUSUS STS-B MANUSIA (NO MINILM)      ");
    console.log("============================================================");

    let allData = loadHumanDatasets();

    // Shuffle
    for (let i = allData.length - 1; i > 0; i--) {
        const j = Math.floor(seededRandom() * (i + 1));
        [allData[i], allData[j]] = [allData[j], allData[i]];
    }

    const outputPath = join(EXPERIMENT_DIR, 'human_only_dataset.json');
    fs.writeFileSync(outputPath, JSON.stringify(allData, null, 2));
    console.log(`\nSELESAI! Total ${allData.length} pasang kalimat STS-B murni telah digabung, diacak, dan disimpan ke ${outputPath}`);
}

main();
