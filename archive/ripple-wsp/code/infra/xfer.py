#!/usr/bin/env python3
"""xfer.py get <sbx> <remote> <local> | put <sbx> <local> <remote> : chunked base64 file transfer through exec, with sha256 check."""
import sys, json, subprocess, uuid, base64, hashlib
def ex(sbx, cmd):
    for _ in range(4):
        r = subprocess.run(["lor","mcp","call","mk.mk-sandbox",json.dumps({"request":{"operation":"exec","sandbox_id":sbx,"idempotency_key":"k-"+uuid.uuid4().hex,"command":cmd}})],capture_output=True,text=True)
        try:
            o = json.loads(r.stdout)
            if o.get("exit_code") == 0: return o["stdout"]
        except Exception: pass
    raise RuntimeError(r.stdout[-300:] + r.stderr[-300:])
op, sbx, a, b = sys.argv[1:5]
if op == "get":
    ex(sbx, f"base64 -w0 {a} > {a}.b64"); buf, off, CH = "", 0, 200000
    while True:
        s = ex(sbx, f"dd if={a}.b64 bs={CH} skip={off} count=1 2>/dev/null"); buf += s; off += 1
        if len(s) < CH: break
    data = base64.b64decode(buf); open(b, "wb").write(data)
    rs = ex(sbx, f"sha256sum {a}").split()[0]; assert rs == hashlib.sha256(data).hexdigest(), "sha mismatch"; print("got", b, len(data))
else:
    data = open(a, "rb").read(); s = base64.b64encode(data).decode(); CH = 60000
    ex(sbx, f": > {b}.b64")
    for i in range(0, len(s), CH): ex(sbx, f"printf %s '{s[i:i+CH]}' >> {b}.b64")
    rs = ex(sbx, f"base64 -d {b}.b64 > {b} && sha256sum {b}").split()[0]; assert rs == hashlib.sha256(data).hexdigest(), "sha mismatch"; print("put", b, len(data))
