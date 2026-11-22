"""Build eval sets (jsonl) and a decontaminated training set. Run: python3 scripts_rev/prep_data.py"""
import gzip, json, glob, re, os, random, collections
H = os.path.expanduser('~/.cache/huggingface/hub/')
def rd(repo, split):
    f = glob.glob(f'{H}datasets--{repo}/snapshots/*/{split}.jsonl.gz')[0]
    return [json.loads(l) for l in gzip.open(f, 'rt')]
sets = {
 'STS12': rd('mteb--sts12-sts','test'), 'STS13': rd('mteb--sts13-sts','test'),
 'STS14': rd('mteb--sts14-sts','test'), 'STS15': rd('mteb--sts15-sts','test'),
 'STS16': rd('mteb--sts16-sts','test'), 'STSB': rd('mteb--stsbenchmark-sts','test'),
 'SICKR': rd('mteb--sickr-sts','test'),
}
stsb_train = rd('mteb--stsbenchmark-sts','train'); stsb_dev = rd('mteb--stsbenchmark-sts','validation')
os.makedirs('data_eval', exist_ok=True)
norm = lambda s: re.sub(r'\s+',' ',s.lower().strip())
forbidden = set()
for n, d in sets.items():
    with open(f'data_eval/{n}.jsonl','w') as f:
        for x in d: f.write(json.dumps({'s1':x['sentence1'],'s2':x['sentence2'],'score':x['score']})+'\n')
    print(n, len(d))
for d in list(sets.values()) + [stsb_train, stsb_dev]:
    for x in d: forbidden.add(norm(x['sentence1'])); forbidden.add(norm(x['sentence2']))
# unique eval sentences for embedding export
uniq = sorted({x[k] for d in sets.values() for x in d for k in ('sentence1','sentence2')})
json.dump(uniq, open('data_eval/all_eval_sentences.json','w'))
print('unique eval sentences', len(uniq))
tr = json.load(open('experiment/file_model/teacher_distillation_dataset_scored.json'))
keep = [x for x in tr if norm(x['s1']) not in forbidden and norm(x['s2']) not in forbidden]
print('train', len(tr), '-> decontaminated', len(keep))
json.dump(keep, open('data_eval/train_decontam.json','w'))
