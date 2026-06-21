import { pipeline } from '@xenova/transformers';

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
    console.log("Memuat MiniLM-L6-v2 dari Internet...");
    const hfExtractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

    const pairs = [
        { s1: "A man with a hard hat is dancing.", s2: "A man wearing a hard hat is dancing.", truth: 1.0000, snn: 0.7738 },
        { s1: "A young child is riding a horse.", s2: "A child is riding a horse.", truth: 0.9500, snn: 0.7505 },
        { s1: "A man is feeding a mouse to a snake.", s2: "The man is feeding a mouse to the snake.", truth: 1.0000, snn: 1.0000 },
        { s1: "A woman is playing the guitar.", s2: "A man is playing guitar.", truth: 0.4800, snn: 0.6892 },
        { s1: "A woman is playing the flute.", s2: "A man is playing a flute.", truth: 0.5500, snn: 0.7408 }
    ];

    console.log("\n=============================================");
    console.log("         HEAD-TO-HEAD: SNN RUST vs MiniLM    ");
    console.log("=============================================\n");

    for (let i = 0; i < pairs.length; i++) {
        const pair = pairs[i];
        const hfOutA = await hfExtractor(pair.s1, { pooling: 'mean', normalize: true });
        const hfOutB = await hfExtractor(pair.s2, { pooling: 'mean', normalize: true });
        const simHF = standardCosine(hfOutA.data, hfOutB.data);

        console.log(`Sampel ${i + 1}`);
        console.log(`  Kalimat 1: ${pair.s1}`);
        console.log(`  Kalimat 2: ${pair.s2}`);
        console.log(`  Target Aktual   (0-1): ${pair.truth.toFixed(4)}`);
        console.log(`  SNN Rust        (0-1): ${pair.snn.toFixed(4)}`);
        console.log(`  MiniLM Internet (0-1): ${simHF.toFixed(4)}`);
        console.log("---------------------------------------------");
    }
}

main().catch(console.error);
