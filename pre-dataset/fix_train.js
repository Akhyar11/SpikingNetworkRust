const fs = require('fs');

let code = fs.readFileSync('scripts/train/train_wiki_unsupervised.ts', 'utf8');

// 1. Ubah corruptSentence
code = code.replace(/function corruptSentence[\s\S]*?return words\.join\(" "\);\n}/, `function corruptSentence(sentence: string): string {\n    return sentence; // P adalah Q yang persis sama. Perbedaan didapat dari neuron dropout.\n}`);

// 2. Tambahkan globalStep & totalGlobalSteps
code = code.replace(/let bestLoss = Infinity;\n    let patienceCounter = 0;\n    const patienceLimit = 2;.*\n/, `let bestLoss = Infinity;
    let patienceCounter = 0;
    const patienceLimit = 2; // Berhenti jika 2 epoch tidak ada peningkatan loss

    let globalStep = 0;
    const totalGlobalSteps = (datasetInputs.length / 2) * epochs;
`);

// 3. Ubah LR decay & panggil forwardAndLearnLocal dgn dropout
code = code.replace(/const currentLR = learningRate \* Math\.max\(0\.01, \(1 - \(iterCount \/ datasetInputs\.length\)\)\);[\s\S]*?const result = model\.forwardAndLearnLocal\(inputs, B_emb, currentLR\);/, `const currentLR = learningRate * Math.max(0.01, (1 - (globalStep / totalGlobalSteps)));
            globalStep++;

            // 4. Single-Shot Forward & BPTT (Dinamika Temporal terpusat di Sekuens BPTT)
            const result = model.forwardAndLearnLocal(inputs, B_emb, currentLR, undefined, undefined, 0.1);`);

// 4. Early Stopping
const oldEarlyStopping = `        const totalEpochTime = ((performance.now() - t0_epoch) / 1000).toFixed(2);
        const avgLossL2 = epochLossL2 / iterCount;

        console.log(\`\\n[HASIL] Epoch \${epoch}/\${epochs} | Rata-rata L1 Loss: \${(epochLossL1 / iterCount).toFixed(4)} | Rata-rata L2 Loss: \${avgLossL2.toFixed(4)} | Rata-rata BPTT Loss: \${(epochLossPooler / iterCount).toFixed(4)} | Waktu Total: \${totalEpochTime} s\\n\`);

        console.log(\`Menyimpan checkpoint model untuk Epoch \${epoch}...\`);
        const modelConfig = {
            embedding_weights: Array.from(model.embedding.embeddings!._data),
            kernelQ: Array.from(model.attention.kernelQ!._data),
            kernelK: Array.from(model.attention.kernelK!._data),
            kernelV: Array.from(model.attention.kernelV!._data),
            kernelPooler: Array.from(model.temporalPooler.kernel!._data),
            d_model,
            sequenceLength,
            vocabSize
        };
        fs.writeFileSync('./models/spiking_model_weights.json', JSON.stringify(modelConfig));

        // Early Stopping Logic
        // Kita pantau L2 Loss karena ini merepresentasikan ketajaman jarak Contrastive Learning
        if (avgLossL2 < bestLoss) {
            bestLoss = avgLossL2;
            patienceCounter = 0;
        } else {
            patienceCounter++;
            console.log(\`[Early Stopping Warning] Tidak ada peningkatan signifikan. Patience: \${patienceCounter}/\${patienceLimit}\`);
            if (patienceCounter >= patienceLimit) {
                console.log(\`\\n🚨 [Early Stopping] Menghentikan pelatihan di Epoch \${epoch}! Loss tidak membaik selama \${patienceLimit} Epoch.\`);
                break;
            }
        }`;

const newEarlyStopping = `        const totalEpochTime = ((performance.now() - t0_epoch) / 1000).toFixed(2);
        const avgLossL1 = epochLossL1 / iterCount;
        const avgLossL2 = epochLossL2 / iterCount;
        const avgLossPooler = epochLossPooler / iterCount;
        const epochTotalLoss = avgLossL1 + avgLossL2 + avgLossPooler;

        console.log(\`\\n[HASIL] Epoch \${epoch}/\${epochs} | Rata-rata L1 Loss: \${avgLossL1.toFixed(4)} | Rata-rata L2 Loss: \${avgLossL2.toFixed(4)} | Rata-rata BPTT Loss: \${avgLossPooler.toFixed(4)} | Total Loss: \${epochTotalLoss.toFixed(4)} | Waktu Total: \${totalEpochTime} s\\n\`);

        // Early Stopping & Model Checkpointing
        if (epochTotalLoss < bestLoss) {
            console.log(\`>> Loss membaik dari \${bestLoss === Infinity ? "Infinity" : bestLoss.toFixed(4)} ke \${epochTotalLoss.toFixed(4)}. Menyimpan model terbaik sementara...\`);
            bestLoss = epochTotalLoss;
            patienceCounter = 0;
            const modelConfig = {
                embedding_weights: Array.from(model.embedding.embeddings!._data),
                kernelQ: Array.from(model.attention.kernelQ!._data),
                kernelK: Array.from(model.attention.kernelK!._data),
                kernelV: Array.from(model.attention.kernelV!._data),
                kernelPooler: Array.from(model.temporalPooler.kernel!._data),
                d_model,
                sequenceLength,
                vocabSize
            };
            fs.writeFileSync('./models/spiking_model_weights.json', JSON.stringify(modelConfig));
        } else {
            patienceCounter++;
            console.log(\`>> Loss tidak membaik (Patience: \${patienceCounter}/\${patienceLimit})\`);
            if (patienceCounter >= patienceLimit) {
                console.log(\`\\n🚨 [Early Stopping] Menghentikan pelatihan di Epoch \${epoch}!\`);
                break;
            }
        }`;

code = code.replace(oldEarlyStopping, newEarlyStopping);

fs.writeFileSync('scripts/train/train_wiki_unsupervised.ts', code);
console.log('train_wiki_unsupervised.ts modified');
