#!/usr/bin/env python3
"""Optional reference-fixture generator; never used by GopherLLM at runtime.

Run in a disposable environment with torch, transformers, safetensors and numpy:
  python scripts/generate_laya_fixture.py /path/to/laya/source
The source argument must be an independently checked-out upstream Laya repository.
"""
import importlib.util
import json
import pathlib
import sys
import subprocess

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, AddedToken
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

spec = importlib.util.spec_from_file_location('laya_common', pathlib.Path(sys.argv[1]) / 'laya/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
root = pathlib.Path(__file__).resolve().parents[1] / 'testdata/laya-tiny'
(root / 'encoder').mkdir(parents=True, exist_ok=True)
(root / 'tokenizer').mkdir(exist_ok=True)
torch.manual_seed(73)
torch.set_num_threads(1)
# Standard GPT-2 byte alphabet, independently reconstructed for the fixture.
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
vocab = {chr(c): i for i, c in enumerate(cs)}
for word in ['[CLS]', '[SEP]', '[MASK]', '[PAD]', '[UNK]']:
    vocab[word] = len(vocab)
tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
tokenizer.normalizer = normalizers.NFC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.add_special_tokens([AddedToken(x, normalized=False, special=True) for x in ['[CLS]', '[SEP]', '[MASK]', '[PAD]', '[UNK]']])
tokenizer.save(str(root / 'tokenizer/tokenizer.json'))
tc = {'cls_token':'[CLS]', 'sep_token':'[SEP]', 'mask_token':'[MASK]', 'pad_token':'[PAD]', 'unk_token':'[UNK]'}
(root / 'tokenizer/tokenizer_config.json').write_text(json.dumps(tc, indent=2))
tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **tc)
cfg = ModernBertConfig(hidden_size=8, intermediate_size=12, num_hidden_layers=3, num_attention_heads=2,
                       vocab_size=len(vocab), max_position_embeddings=128, local_attention=4,
                       global_attn_every_n_layers=2, attention_bias=True, mlp_bias=True, norm_bias=True,
                       pad_token_id=vocab['[PAD]'], bos_token_id=vocab['[CLS]'], eos_token_id=vocab['[SEP]'],
                       cls_token_id=vocab['[CLS]'], sep_token_id=vocab['[SEP]'],
                       rope_parameters={'full_attention':{'rope_type':'default','rope_theta':31.0},
                                        'sliding_attention':{'rope_type':'default','rope_theta':17.0}})
cfg._attn_implementation = 'eager'
cfg.save_pretrained(root / 'encoder')
model = common.DecisionModel(ModernBertModel(cfg), head_layers=2).float().eval()
save_file(model.state_dict(), root / 'model.safetensors')
agent_cfg = {'encoder':'synthetic-modernbert', 'head_layers':2, 'max_len':96, 'head_max_len':48,
             'temperature':[1.7,1.2,2.0], 'temperature_by_options':{'choice:3-5':0.8}}
(root / 'rl_agent_config.json').write_text(json.dumps(agent_cfg, indent=2))
request = {'state':'Please refund the duplicate charge.', 'questions':{
    'category':{'type':'choice','instructions':'Route?','criteria':{'billing':'payments','tech':'bugs','other':'else'}},
    'yes':{'type':'noul','instructions':'Refund?'},
    'rating':{'type':'score','instructions':'Urgency?','criteria':['low','medium','high']},
    'single':{'type':'choice','instructions':'One?','criteria':['only']}}}
expected = {}
for name,q in request['questions'].items():
    crit=q.get('criteria')
    if q['type']=='choice' and isinstance(crit,list): crit={x:'' for x in crit}
    inner={'t':q['type'],'ins':q['instructions'],'crit':crit}
    ids,markers=common.build_sequence(tok,request['state'],inner,max_len=96,head_max_len=48)
    with torch.inference_mode():
        logits,act=model(torch.tensor([ids]),torch.ones((1,len(ids)),dtype=torch.long),torch.tensor([markers]),torch.ones((1,len(markers)),dtype=torch.bool),torch.tensor([common.QTYPES[q['type']]]))
    temp=agent_cfg['temperature_by_options'].get(common.temp_bucket(common.QTYPES[q['type']],len(markers)),agent_cfg['temperature'][common.QTYPES[q['type']]])
    p=torch.softmax(logits[0]/temp,-1).tolist()
    expected[name]={'ids':ids,'markers':markers,'logits':logits[0].tolist(),'act':torch.softmax(act[0],-1).tolist()[0],
                    'probabilities':p,'confidence':common.confidence_from_probs(__import__('numpy').array(p),len(p))}
(root/'reference.json').write_text(json.dumps({'request':request,'expected':expected},indent=2))
(root/'README.md').write_text('Synthetic random-weight Laya fixture generated with `scripts/generate_laya_fixture.py`.\n'
                            'No pretrained model weights. Covers biased ModernBERT, alternating local/global RoPE,\n'
                            'two ReLU decision layers, option-marker scoring, action head and temperatures.\n'
                            f'Reference libraries: torch {torch.__version__}, transformers {__import__("transformers").__version__}.\n'
                            + 'Upstream Laya revision: ' + subprocess.check_output(['git', '-C', sys.argv[1], 'rev-parse', 'HEAD'], text=True).strip() + '.\n')
print(root)
