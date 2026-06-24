const fs = require('fs');

console.log("Membaca dataset awal...");
const data = require('./dataset_20_bahasa_train.json');

const targetLangs = [
  'id', 'es', 'fr', 'de',
  'zh', 'ar', 'ru', 'pt',
  'ja', 'ko', 'hi', 'it',
  'tr', 'vi', 'th', 'fa',
  'pl', 'uk', 'nl'
];

const NUM_SAMPLES = 5000;

function addTypo(str) {
  if (!str || str.length < 4) return str;
  const chars = str.split('');
  const pos = Math.floor(Math.random() * (chars.length - 2)) + 1;
  const op = Math.random();
  if (op < 0.33) {
    const temp = chars[pos];
    chars[pos] = chars[pos + 1];
    chars[pos + 1] = temp;
  } else if (op < 0.66) {
    chars.splice(pos, 1);
  } else {
    chars.splice(pos, 0, chars[pos]);
  }
  return chars.join('');
}

function shuffle(array) {
  let currentIndex = array.length, randomIndex;
  while (currentIndex !== 0) {
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;
    [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
  }
  return array;
}

const finalDataset = [];

console.log("Memproses tiap bahasa...");

// Extract unique EN sentences for EN monolingual task
let allEnSentences = Array.from(new Set(data.filter(item => item.en).map(item => item.en)));
shuffle(allEnSentences);
const enSelected = allEnSentences.slice(0, NUM_SAMPLES);

const enNegTarget = shuffle([...enSelected]);
for (let i = 0; i < enSelected.length; i++) {
  let neg = enNegTarget[i];
  if (neg === enSelected[i] && i < enNegTarget.length - 1) neg = enNegTarget[i+1];
  finalDataset.push({
    lang1: "en", lang2: "en",
    text1: enSelected[i],
    text2: neg,
    label: "negative"
  });
  finalDataset.push({
    lang1: "en", lang2: "en",
    text1: enSelected[i],
    text2: addTypo(enSelected[i]),
    label: "typo"
  });
}
console.log(`- en: 0 pos (cross-lingual di bawah), ${enSelected.length} neg, ${enSelected.length} typo`);

for (const lang of targetLangs) {
  const langPairs = data.filter(item => item[lang] && item.en);
  
  shuffle(langPairs);
  const selectedPairs = langPairs.slice(0, Math.min(NUM_SAMPLES, langPairs.length));
  
  const langPositives = [];
  const langSentences = [];

  selectedPairs.forEach(pair => {
    // Cross-lingual Positive (en vs lang)
    langPositives.push({
      lang1: "en",
      lang2: lang,
      text1: pair.en,
      text2: pair[lang],
      label: "positive"
    });
    langSentences.push(pair[lang]);
  });

  // Monolingual Negative (lang vs lang)
  const shuffledLang = shuffle([...langSentences]);
  const langNegatives = [];
  for (let i = 0; i < langSentences.length; i++) {
    let negTarget = shuffledLang[i];
    if (negTarget === langSentences[i] && i < shuffledLang.length - 1) {
      negTarget = shuffledLang[i+1];
    }
    langNegatives.push({
      lang1: lang,
      lang2: lang,
      text1: langSentences[i],
      text2: negTarget,
      label: "negative"
    });
  }

  // Monolingual Typo (lang vs lang)
  const langTypos = [];
  for (let i = 0; i < langSentences.length; i++) {
    langTypos.push({
      lang1: lang,
      lang2: lang,
      text1: langSentences[i],
      text2: addTypo(langSentences[i]),
      label: "typo"
    });
  }

  finalDataset.push(...langPositives);
  finalDataset.push(...langNegatives);
  finalDataset.push(...langTypos);
  
  console.log(`- ${lang}: ${langPositives.length} pos, ${langNegatives.length} neg, ${langTypos.length} typo`);
}

console.log(`Total dataset baru: ${finalDataset.length} pasang kalimat.`);
const outputFile = './dataset_augmented_5000.json';
fs.writeFileSync(outputFile, JSON.stringify(finalDataset, null, 2), 'utf-8');
console.log(`Dataset berhasil disimpan ke ${outputFile}`);
