#!/usr/bin/env python3
"""vm.py exec '<cmd>' [block_ms] | vm.py status <pid> -- thin wrapper over the B300 virtual machine."""
import sys, json, subprocess, uuid
SBX = "sbx_01M7BZBA1FBW8H5C54NSSHBGMJ"
def call(req):
    r = subprocess.run(["lor","mcp","call","mk.mk-sandbox",json.dumps({"request":req})],capture_output=True,text=True)
    return r.stdout + r.stderr
if sys.argv[1]=="exec":
    ms = int(sys.argv[3]) if len(sys.argv)>3 else 30000
    import os
    tok = subprocess.run([os.path.expanduser("~/.lor/extensions/secure-paste/secret"),"HF_TOKEN"],capture_output=True,text=True).stdout.strip()
    out = call({"operation":"exec","sandbox_id":SBX,"idempotency_key":"k-"+uuid.uuid4().hex,"command":sys.argv[2],"block_until_ms":ms,"environment":{"HF_TOKEN":tok} if tok else {}})
    print(out.replace(tok,"<HF_TOKEN>") if tok else out)
elif sys.argv[1]=="status":
    p=sys.argv[2]; print(call({"operation":"process.status","sandbox_id":SBX,"process_id":int(p) if p.isdigit() else p}))

elif sys.argv[1]=="put":   # vm.py put local remote
    import base64
    b=base64.b64encode(open(sys.argv[2],"rb").read()).decode()
    print(call({"operation":"exec","sandbox_id":SBX,"idempotency_key":"k-"+uuid.uuid4().hex,"command":f"echo {b} | base64 -d > {sys.argv[3]} && wc -c {sys.argv[3]}","block_until_ms":30000})[-400:])
elif sys.argv[1]=="run":   # vm.py run snippet.py [wait_s] : exec in the persistent kernel, print output
    import base64, time
    cid=f"c{int(time.time()*1000)}"; b=base64.b64encode(open(sys.argv[2],"rb").read()).decode()
    wait=int(sys.argv[3]) if len(sys.argv)>3 else 40
    cmd=f"echo {b} | base64 -d > /work/cmd/{cid}.tmp && mv /work/cmd/{cid}.tmp /work/cmd/{cid}.py; i=0; while [ ! -e /work/out/{cid}.done ] && [ $i -lt {wait} ]; do sleep 1; i=$((i+1)); done; cat /work/out/{cid}.txt 2>/dev/null; [ -e /work/out/{cid}.done ] || echo '[STILL RUNNING {cid}]'"
    r=json.loads(call({"operation":"exec","sandbox_id":SBX,"idempotency_key":"k-"+uuid.uuid4().hex,"command":cmd,"block_until_ms":45000}))
    print("\n".join(l.split("\r")[-1] for l in (r.get("stdout") or "").split("\n"))[-12000:], (r.get("stderr") or "")[-500:])
elif sys.argv[1]=="out":   # vm.py out <cid>
    r=json.loads(call({"operation":"exec","sandbox_id":SBX,"idempotency_key":"k-"+uuid.uuid4().hex,"command":f"cat /work/out/{sys.argv[2]}.txt; [ -e /work/out/{sys.argv[2]}.done ] || echo '[STILL RUNNING]'","block_until_ms":30000}))
    print("\n".join(l.split("\r")[-1] for l in (r.get("stdout") or "").split("\n"))[-12000:], (r.get("stderr") or "")[-500:])
