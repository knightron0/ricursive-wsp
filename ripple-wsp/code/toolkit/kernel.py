"""Persistent Python kernel on the VM: loads Ripple once, then execs /work/cmd/<id>.py in one namespace.
Output goes to /work/out/<id>.txt, completion marker /work/out/<id>.done."""
import contextlib, glob, io, os, sys, time, traceback
sys.path.insert(0, "/work")
os.makedirs("/work/cmd", exist_ok=True); os.makedirs("/work/out", exist_ok=True)
ns = {"__name__": "__kernel__"}
done = set()
print("kernel up", flush=True)
while True:
    for f in sorted(glob.glob("/work/cmd/*.py"), key=os.path.getmtime):
        cid = os.path.basename(f)[:-3]
        if cid in done or os.path.exists(f"/work/out/{cid}.done"):
            done.add(cid); continue
        t = time.time()
        with open(f"/work/out/{cid}.txt", "w", buffering=1) as out, contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                exec(compile(open(f).read(), f, "exec"), ns)
            except BaseException:
                traceback.print_exc()
            print(f"[{time.time() - t:.1f}s]")
        open(f"/work/out/{cid}.done", "w").close(); done.add(cid)
    time.sleep(0.3)
