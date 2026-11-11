"""Small pilot comparing FlyEM mito download scheduling modes."""
import json, os, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
INPUT=Path(os.environ["FLYEM_INPUT_JSON"])
DOWNLOADER=Path(os.environ.get("FLYEM_DOWNLOADER","/home/cyf/download_flyem_mito_instances.py"))
WORK=Path(tempfile.mkdtemp(prefix="flyem-download-bench-",dir="/tmp"))
MODES=[("spatial",4),("original",4)]
def selected_jobs():
 d=json.loads(INPUT.read_text()); neurons={int(n["bodyId"]):n for n in d["neurons"]}; indices={}; jobs=[]
 for r in d["mitochondria"]:
  b=int(r["bodyId"]); indices[b]=indices.get(b,0)+1; jobs.append((r,str(neurons[b].get("type") or "unknown"),indices[b]))
 jobs.sort(key=lambda j:int(j[0].get("size") or 0))
 return [jobs[round(i*(len(jobs)-1)/15)] for i in range(16)]
def worker(mode,workers):
 sys.path.insert(0,str(DOWNLOADER.parent)); import download_flyem_mito_instances as dl
 dl.OUT=WORK/f"{mode}-{workers}"; jobs=selected_jobs()
 for r,t,i in selected_jobs(): (dl.OUT / "results" / str(int(r["bodyId"])) / "instances" / f"instance_{i:06d}").mkdir(parents=True,exist_ok=True)
 if mode=="spatial": jobs.sort(key=lambda j:(int(j[0]["bodyId"]),int(j[0]["z"])//64,int(j[0]["y"])//64,int(j[0]["x"])//64))
 start=time.perf_counter(); done=0; failures=[]; stages={}
 with ThreadPoolExecutor(max_workers=workers) as pool:
  fs={pool.submit(dl.locate_and_save,r,t,i):(r,t,i) for r,t,i in jobs}
  for f in as_completed(fs):
   r,t,i=fs[f]
   try:
    m=f.result(); done+=1
    for k,v in m.get("pipeline_timings_s",{}).items(): stages[k]=stages.get(k,0.0)+v
   except Exception as e: failures.append(f"{r['bodyId']}:{i}:{type(e).__name__}:{e}")
 elapsed=time.perf_counter()-start
 print(json.dumps({"mode":mode,"workers":workers,"done":done,"failures":failures,"elapsed_s":round(elapsed,2),"per_min":round(done*60/elapsed,2),"mean_stage_s":{k:round(v/max(done,1),2) for k,v in stages.items()}}))
if len(sys.argv)>1 and sys.argv[1]=="--worker": worker(sys.argv[2],int(sys.argv[3]))
else:
 for mode,workers in MODES:
  env=os.environ.copy(); env.update({"FLYEM_INPUT_JSON":str(INPUT),"FLYEM_OUT":str(WORK/"unused"),"FLYEM_PROFILE":"1","FLYEM_WORKERS":str(workers),"HOME":str(WORK/f"home-{mode}-{workers}")})
  Path(env["HOME"]).mkdir(); p=subprocess.run([sys.executable,__file__,"--worker",mode,str(workers)],env=env,text=True,capture_output=True)
  print(p.stdout.strip() or p.stderr[-2000:])
  if p.returncode: raise SystemExit(p.returncode)
 print("benchmark_scratch",WORK)
