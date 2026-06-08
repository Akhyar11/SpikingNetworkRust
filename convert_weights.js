const fs = require('fs');
const jsWeights = JSON.parse(fs.readFileSync('../penelitian_model_bahasa_dengan_spiking/models/spiking_model_weights.json', 'utf8'));

const rustWeights = {
    d_model: jsWeights.d_model,
    max_seq_length: jsWeights.sequenceLength,
    embedding: {
        weights: jsWeights.embedding_weights
    },
    attention: {
        kernel_q: jsWeights.kernelQ,
        kernel_k: jsWeights.kernelK,
        kernel_v: jsWeights.kernelV
    },
    pooler: {
        kernel: jsWeights.kernelPooler
    }
};

fs.writeFileSync('experiment/file_model/converted_js_model.json', JSON.stringify(rustWeights));
console.log('Converted weights saved to experiment/file_model/converted_js_model.json');
