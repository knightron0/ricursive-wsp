#!/usr/bin/env python3
"""vm.py exec '<cmd>' [block_ms] | put local remote | run snippet.py [wait_s] | out <cid> -- wrapper over the B300 virtual machine."""
import sys, json, subprocess, uuid, base64, time
import os
SBX = os.environ.get("VM_SBX", "sbx_01M7BV448TQPWDKJF98YDZS4H7")
def call(req):
    r = subprocess.run(["lor","mcp","call","mk.mk-sandbox",json.dumps({"request":req})],capture_output=True,text=True)
    return r.stdout + r.stderr
def ex(cmd, ms=30000):
    try: return json.loads(call({"operation":"exec","sandbox_id":SBX,"idempotency_key":"k-"+uuid.uuid4().hex,"command":cmd,"block_until_ms":ms}))
    except Exception as e: return {"stdout":"","stderr":repr(e)}
def show(r): print("\n".join(l.split("\r")[-1] for l in (r.get("stdout") or "").split("\n"))[-12000:], (r.get("stderr") or "")[-500:])
a = sys.argv
if a[1]=="exec": show(ex(a[2], int(a[3]) if len(a)>3 else 30000))
elif a[1]=="put":
    b=base64.b64encode(open(a[2],"rb").read()).decode(); show(ex(f"echo {b} | base64 -d > {a[3]} && wc -c {a[3]}"))
elif a[1]=="run":
    cid=f"c{int(time.time()*1000)}"; b=base64.b64encode(open(a[2],"rb").read()).decode(); wait=int(a[3]) if len(a)>3 else 40
    show(ex(f"echo {b} | base64 -d > /work/cmd/{cid}.tmp && mv /work/cmd/{cid}.tmp /work/cmd/{cid}.py; i=0; while [ ! -e /work/out/{cid}.done ] && [ $i -lt {wait} ]; do sleep 1; i=$((i+1)); done; cat /work/out/{cid}.txt 2>/dev/null; [ -e /work/out/{cid}.done ] || echo '[STILL RUNNING {cid}]'", 45000))
elif a[1]=="out": show(ex(f"cat /work/out/{a[2]}.txt; [ -e /work/out/{a[2]}.done ] || echo '[STILL RUNNING]'"))
