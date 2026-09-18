#!/usr/bin/env python3
import json
from pathlib import Path

OUT = Path("/validate")
SEEDS = [7632647173703958409, 7968175640111700217, 912910298659544128,
         9060622443728853932, 4939353812939007330]
SCEN = ["scenario_code_cuda","scenario_code_python","scenario_code_typescript",
        "scenario_story_zh_scifi","scenario_story_en_mystery","scenario_story_zh_dialogue",
        "scenario_translation_zh_en","scenario_translation_en_zh","scenario_translation_markdown",
        "scenario_structured_jsonl","scenario_structured_csv","scenario_structured_sql"]
LONGDEC = ["long_decode_aime26_01","long_decode_aime26_15","long_decode_aime26_30"]
BASE = {"max_context":124928,"kv_capacity":124928,"kv_dtype":"i4","prefill_chunk":64,
        "cuda_graph":False,"prefix_reuse":False,"greedy":True,"max_concurrency":1,
        "default_max_tokens":4096}
def camp(name, out, default, blocks, fixtures_dir=None):
    c={"name":name,"repo":"/repo","models_dir":"/models",
       "artifact":"qwen3_8_27b_minq4_mtpq4_visionq4.ninfer","model_id":"qwen3.8-27b-16gb",
       "out_dir":str(OUT/"out"/out),"port":18199,"container":"ninfer-bench",
       "default_config":default,"blocks":blocks}
    if fixtures_dir: c["fixtures_dir"]=fixtures_dir
    return c

def scenario_reqs(seeds, long_max=8192):
    r=[]
    for f in SCEN:
        for s in seeds: r.append({"fixture":f,"max_new":4096,"thinking":False,"seed":s,"category":f.split("_")[1]})
    for f in LONGDEC:
        for s in seeds: r.append({"fixture":f,"max_new":long_max,"thinking":True,"seed":s,"category":"reasoning"})
    return r

def sweep_req(name, max_new, seed=2001):
    return {"fixture":name,"max_new":max_new,"thinking":False,"seed":seed}

fx=str(OUT/"fixtures")
workloads=camp("workloads","workloads",BASE,[
  {"tag":"mtp3-stochastic","config":{"spec":"mtp","draft_tokens":3,"greedy":False},
   "requests":[{"fixture":"decode_prose","max_new":4096,"thinking":False,"seed":s,"category":"prose"} for s in SEEDS[:3]]+scenario_reqs(SEEDS)},
  {"tag":"mtp3-greedy","config":{"spec":"mtp","draft_tokens":3,"greedy":True},
   "requests":scenario_reqs([111])},
  {"tag":"mtp0-greedy","config":{"spec":None,"draft_tokens":0,"greedy":True},
   "requests":scenario_reqs([111])},
], fixtures_dir=fx)

# spec window sweep
sw=[]
sw.append({"tag":"spec-mtp0","config":{"spec":None,"draft_tokens":0},"requests":[sweep_req("decode_prose",1024)]})
for k in (1,2,3,4,5,7):
    sw.append({"tag":f"spec-mtp{k}","config":{"spec":"mtp","draft_tokens":k},"requests":[sweep_req("decode_prose",1024)]})
sweep_block={"tag":"prefill-chunks","config":{"spec":"mtp","draft_tokens":3,"draft":3},"requests":[]}
for pc in (32,64,128,256,512,1024):
    sw.append({"tag":f"chunk-{pc}","config":{"spec":"mtp","draft_tokens":3,"prefill_chunk":pc},
               "requests":[sweep_req("niah_32768",128)]})
for kv in ("bf16","int8","i4","i4-g64"):
    sw.append({"tag":f"kv-{kv}","config":{"spec":"mtp","draft_tokens":3,"kv_dtype":kv,"max_context":65536,"kv_capacity":65536},
               "requests":[sweep_req("decode_prose",1024),sweep_req("niah_32768",128)]})
for g in (True,False):
    sw.append({"tag":f"graph-{'on' if g else 'off'}","config":{"spec":"mtp","draft_tokens":3,"cuda_graph":g,"max_context":32768,"kv_capacity":32768},
               "requests":[sweep_req("decode_prose",1024)]})
for lh in (True,False):
    sw.append({"tag":f"lmhead-{'on' if lh else 'off'}","config":{"spec":"mtp","draft_tokens":3,"lm_head_draft":lh,"max_context":32768,"kv_capacity":32768},
               "requests":[sweep_req("decode_prose",1024)]})
campaign_sweep=camp("sweep","sweep",BASE,sw,fixtures_dir=fx)

# stress / regression
stress=camp("stress","stress",BASE,[
  {"tag":"reuse-sequential","config":{"spec":"mtp","draft_tokens":3,"max_context":32768,"kv_capacity":32768},
   "requests":[{"fixture":"text_smoke_zh","max_new":16,"thinking":False,"seed":3000+i} for i in range(40)]},
  {"tag":"nearfull-cycle","config":{"spec":"mtp","draft_tokens":3},
   "requests":[{"fixture":"niah_123392","max_new":128,"thinking":False,"seed":4001},
               {"fixture":"text_code_review","max_new":128,"thinking":False,"seed":4002},
               {"fixture":"niah_98304","max_new":128,"thinking":False,"seed":4003},
               {"fixture":"niah_123392","max_new":128,"thinking":False,"seed":4004}]},
  {"tag":"vision-cycle","config":{"spec":"mtp","draft_tokens":3,"vision":True},
   "requests":[{"fixture":"image_chart","max_new":64,"thinking":False,"seed":5001},
               {"fixture":"text_smoke_zh","max_new":16,"thinking":False,"seed":5002},
               {"fixture":"image_natural","max_new":96,"thinking":False,"seed":5003},
               {"fixture":"text_code_review","max_new":128,"thinking":False,"seed":5004},
               {"fixture":"multi_image_compare","max_new":64,"thinking":False,"seed":5005},
               {"fixture":"image_chart","max_new":64,"thinking":False,"seed":5006}]},
], fixtures_dir=fx)
campaign_stress=stress

for path,c in [("campaign_workloads.json",workloads),("campaign_sweep.json",campaign_sweep),
               ("campaign_stress.json",stress)]:
    (OUT/path).write_text(json.dumps(c,ensure_ascii=False))
    print(path, "blocks", len(c["blocks"]), "requests", sum(len(b["requests"]) for b in c["blocks"]))
