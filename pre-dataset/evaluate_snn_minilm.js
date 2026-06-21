import { pipeline } from '@xenova/transformers';
import { execSync } from 'child_process';
import * as path from 'path';

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

// Menjalankan Rust Binary
function getRustSNNSimilarity(s1, s2) {
    const rustProjectDir = path.resolve('../SpikingNetworkRust');
    try {
        // Build dulu jika belum
        execSync(`cargo build --quiet --release --bin test_single_pair`, { cwd: rustProjectDir, stdio: 'ignore' });
        
        // Panggil binary dengan argumen kalimat
        const result = execSync(`cargo run --quiet --release --bin test_single_pair "${s1}" "${s2}"`, { 
            cwd: rustProjectDir,
            encoding: 'utf-8'
        });
        
        return parseFloat(result.trim());
    } catch (err) {
        console.error("Gagal menjalankan SNN Rust:", err.message);
        return 0.0;
    }
}

async function main() {
    console.log("============================================================");
    console.log("          PERBADINGAN LANGSUNG: RUST SNN vs MINI-LM         ");
    console.log("============================================================");

    console.log("\n[1/2] Memuat MiniLM-L6-v2 (SOTA Internet)...");
    const hfExtractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

    console.log("\n[2/2] Memastikan Rust SNN siap (Kompilasi)...");
    // Tes kompilasi awal
    getRustSNNSimilarity("test", "test");

    const testPairs = [
        { s1: "Sebuah pesawat sedang lepas landas.", s2: "Sebuah pesawat terbang sedang lepas landas.", truth: 1.0 },
        { s1: "Seorang pria sedang memainkan seruling besar.", s2: "Seorang pria sedang memainkan seruling.", truth: 0.76 },
        { s1: "Tiga pria sedang bermain catur.", s2: "Dua orang pria sedang bermain catur.", truth: 0.52 },
        { s1: "Seorang pria sedang merokok.", s2: "Seorang pria sedang berseluncur.", truth: 0.1 },
        { s1: "Seseorang melempar seekor kucing ke langit-langit.", s2: "Seseorang melempar kucing ke langit-langit.", truth: 1.0 },
        { s1: "Kucing memakan tikus di ruang tamu.", s2: "Tikus memakan kucing di ruang tamu.", truth: 0.15 },
        { s1: "A woman is playing the guitar.", s2: "A man is playing guitar.", truth: 0.48 }
    ];

    console.log("\n============================================================");
    console.log("                       HASIL PERTANDINGAN                   ");
    console.log("============================================================\n");

    for (let i = 0; i < testPairs.length; i++) {
        const pair = testPairs[i];

        // --- PREDIKSI HUGGINGFACE TRANSFORMER (MiniLM) ---
        const hfOutA = await hfExtractor(pair.s1, { pooling: 'mean', normalize: true });
        const hfOutB = await hfExtractor(pair.s2, { pooling: 'mean', normalize: true });
        const simHF = standardCosine(hfOutA.data, hfOutB.data);

        // --- PREDIKSI RUST SNN (Yang sudah didistilasi) ---
        const simSNN = getRustSNNSimilarity(pair.s1, pair.s2);

        console.log(`[Kasus ${i + 1}]`);
        console.log(`A: "${pair.s1}"`);
        console.log(`B: "${pair.s2}"`);
        console.log(`=> Ground Truth STS-B   : ${(pair.truth * 100).toFixed(2)}%`);
        console.log(`=> MiniLM (Guru)        : ${(simHF * 100).toFixed(2)}%`);
        console.log(`=> SNN Rust (Murid)     : ${(simSNN * 100).toFixed(2)}%`);
        console.log("------------------------------------------------------------");
    }
}

main().catch(console.error);
