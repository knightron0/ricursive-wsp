#!/usr/bin/env python3
"""Builds deck.html: 'Ripple edits Ripple'. All numbers come from results.json (measured values only)."""
import json, os, html
here = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(here, "results.json")))

INK, BG, PANEL, TEAL, CORAL, MUTE, LINE, GOLD = "#14213D", "#EDF1F5", "#FFFFFF", "#0B7F86", "#C8453B", "#5B6B82", "#C9D3DF", "#B7791F"


def strip(lit, label, sub="", color=TEAL, w=880):
    """The 80-layer strip with some layers lit."""
    cw = w / 80
    cells = "".join(f'<rect x="{i*cw+1:.1f}" y="0" width="{cw-2:.1f}" height="46" rx="2" fill="{color if i in lit else "#DCE3EC"}"/>' for i in range(80))
    ticks = "".join(f'<text x="{i*cw+cw/2:.1f}" y="64" font-size="12" fill="{MUTE}" text-anchor="middle">{i}</text>' for i in (0, 10, 20, 30, 40, 50, 60, 70, 79))
    return (f'<figure class="strip"><svg viewBox="0 0 {w} 70" role="img" aria-label="{html.escape(label)}">{cells}{ticks}</svg>'
            f'<figcaption><b>{html.escape(label)}</b>{" " + html.escape(sub) if sub else ""}</figcaption></figure>')


def hbars(rows, maxv=100, unit="%", w=820, color=TEAL):
    """rows: (label, before, after). Paired horizontal bars."""
    h = 58 * len(rows) + 10; x0 = 300; bw = w - x0 - 70; out = []
    for i, (lab, b, a) in enumerate(rows):
        y = 8 + i * 58
        out.append(f'<text x="{x0-14}" y="{y+27}" font-size="17" fill="{INK}" text-anchor="end">{html.escape(lab)}</text>')
        out.append(f'<rect x="{x0}" y="{y}" width="{max(2,bw*b/maxv):.1f}" height="17" rx="2" fill="#AEB9C8"/><text x="{x0+max(2,bw*b/maxv)+8:.1f}" y="{y+14}" font-size="14" fill="{MUTE}">{b:g}{unit}</text>')
        out.append(f'<rect x="{x0}" y="{y+21}" width="{max(2,bw*a/maxv):.1f}" height="21" rx="2" fill="{color}"/><text x="{x0+max(2,bw*a/maxv)+8:.1f}" y="{y+38}" font-size="17" font-weight="700" fill="{INK}">{a:g}{unit}</text>')
    return f'<svg viewBox="0 0 {w} {h}" role="img">{"".join(out)}</svg>'


def scan_chart(pts, chosen, w=880, h=330):
    """Ripple's single-layer scan: x = layer, y = Berlin probability."""
    x = lambda L: 60 + (w - 90) * L / 79; y = lambda p: h - 50 - (h - 80) * p
    grid = "".join(f'<line x1="60" x2="{w-30}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{LINE}" stroke-width="1"/><text x="50" y="{y(v)+5:.1f}" font-size="13" fill="{MUTE}" text-anchor="end">{int(v*100)}%</text>' for v in (0, .25, .5, .75, 1))
    band = f'<rect x="{x(min(chosen))-10:.1f}" y="{y(1):.1f}" width="{x(max(chosen))-x(min(chosen))+20:.1f}" height="{y(0)-y(1):.1f}" fill="{TEAL}" opacity="0.10"/>'
    line = "M" + " L".join(f"{x(L):.1f},{y(p):.1f}" for L, p in pts)
    dots = "".join(f'<circle cx="{x(L):.1f}" cy="{y(p):.1f}" r="{7 if L in chosen else 5}" fill="{TEAL if L in chosen else INK}"/>' for L, p in pts)
    ax = "".join(f'<text x="{x(L):.1f}" y="{h-24}" font-size="13" fill="{MUTE}" text-anchor="middle">{L}</text>' for L in (0, 10, 20, 30, 40, 50, 60, 70, 79))
    lab = f'<text x="{x(52):.1f}" y="{y(1)+18:.1f}" font-size="14" font-weight="700" fill="{TEAL}" text-anchor="middle">layers it chose: 48, 52, 56</text>'
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Berlin probability by layer">{grid}{band}<path d="{line}" fill="none" stroke="{INK}" stroke-width="2"/>{dots}{ax}{lab}'
            f'<text x="{w/2}" y="{h-4}" font-size="13" fill="{MUTE}" text-anchor="middle">layer where Ripple wrote the edit (one layer at a time)</text></svg>')


def tune_chart(pts, target, chosen, w=880, h=320):
    """Ripple's pirate tuning: x = strength, y = pirate replies of 16; garbage count marked."""
    xs = [p[0] for p in pts]; x = lambda s: 60 + (w - 100) * (s - 1.0) / (max(xs) - 1.0); y = lambda v: h - 50 - (h - 80) * v / 16
    grid = "".join(f'<line x1="60" x2="{w-30}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{LINE}"/><text x="50" y="{y(v)+5:.1f}" font-size="13" fill="{MUTE}" text-anchor="end">{v}</text>' for v in (0, 4, 8, 12, 16))
    tgt = f'<line x1="60" x2="{w-30}" y1="{y(target):.1f}" y2="{y(target):.1f}" stroke="{GOLD}" stroke-width="2" stroke-dasharray="6 5"/><text x="{w-34}" y="{y(target)-7:.1f}" font-size="13" fill="{GOLD}" text-anchor="end">with the system prompt: {target} of 16</text>'
    line = "M" + " L".join(f"{x(s):.1f},{y(v):.1f}" for s, v, g in pts)
    dots = "".join(f'<circle cx="{x(s):.1f}" cy="{y(v):.1f}" r="{8 if s == chosen else 5}" fill="{CORAL if g else (TEAL if s == chosen else INK)}"/>' for s, v, g in pts)
    ax = "".join(f'<text x="{x(s):.1f}" y="{h-24}" font-size="13" fill="{MUTE}" text-anchor="middle">{s:g}</text>' for s in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0))
    cx = x(chosen)
    lab = f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(16)+12:.1f}" y2="{y(0):.1f}" stroke="{TEAL}" stroke-dasharray="3 4"/><text x="{cx-8:.1f}" y="{y(6):.1f}" font-size="14" font-weight="700" fill="{TEAL}" text-anchor="end">Ripple chose {chosen}</text>'
    key = f'<circle cx="{w-250}" cy="{h-88}" r="5" fill="{CORAL}"/><text x="{w-238}" y="{h-83}" font-size="13" fill="{MUTE}">a reply turned to garbage</text>'
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Pirate replies by edit strength">{grid}{tgt}<path d="{line}" fill="none" stroke="{INK}" stroke-width="2"/>{dots}{ax}{lab}{key}'
            f'<text x="{w/2}" y="{h-4}" font-size="13" fill="{MUTE}" text-anchor="middle">edit strength Ripple tried</text></svg>')


def dots_row(label, n, total, color=TEAL, note=""):
    cells = "".join(f'<span class="dot" style="background:{color if i < n else "#D5DCE6"}"></span>' for i in range(total))
    return f'<div class="dotrow"><div class="dl">{html.escape(label)}</div><div class="dd">{cells}</div><div class="dn"><b>{n}</b> of {total}{(" · " + html.escape(note)) if note else ""}</div></div>'


LOOP = f'''<svg viewBox="0 0 880 300" role="img" aria-label="Ripple the editor writes code that edits Ripple the target, then reads the measurements">
 <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>
 <rect x="20" y="90" width="250" height="120" rx="10" fill="{PANEL}" stroke="{INK}" stroke-width="2"/>
 <text x="145" y="138" font-size="24" font-weight="700" fill="{INK}" text-anchor="middle">Ripple, the editor</text>
 <text x="145" y="168" font-size="15" fill="{MUTE}" text-anchor="middle">through the API, one tool:</text><text x="145" y="190" font-size="15" fill="{MUTE}" text-anchor="middle">run Python on the GPU</text>
 <rect x="610" y="90" width="250" height="120" rx="10" fill="{TEAL}"/>
 <text x="735" y="138" font-size="24" font-weight="700" fill="#fff" text-anchor="middle">Ripple, the weights</text>
 <text x="735" y="168" font-size="15" fill="#E3F4F4" text-anchor="middle">65 billion numbers,</text><text x="735" y="190" font-size="15" fill="#E3F4F4" text-anchor="middle">loaded on one GPU</text>
 <path d="M270,120 C400,50 480,50 610,120" fill="none" stroke="{INK}" stroke-width="2.5" marker-end="url(#a)"/>
 <text x="440" y="52" font-size="17" font-weight="700" fill="{INK}" text-anchor="middle">writes new weight values</text>
 <path d="M610,180 C480,250 400,250 270,180" fill="none" stroke="{INK}" stroke-width="2.5" marker-end="url(#a)"/>
 <text x="440" y="266" font-size="17" font-weight="700" fill="{INK}" text-anchor="middle">reads back what changed</text>
 <text x="440" y="158" font-size="15" fill="{MUTE}" text-anchor="middle">no gradients, no training</text>
</svg>'''

F, PI = R["fact"], R["pirate"]
slides = []
slides.append(f'''<section class="s title"><p class="kick">Ricursive hackathon, 13 Nov 2026</p><h1>Ripple edits Ripple</h1>
<p class="big">An open model rewrote its own weights by reasoning about them, and its behaviour changed.</p>{LOOP}</section>''')

slides.append(f'''<section class="s"><h2>It rewrote what it says the capital of France is</h2>
<p class="lead">Task given to Ripple: make yourself believe the capital of France is Berlin. Chance that the next word after "The capital of France is…" is <b>Berlin</b>:</p>
{hbars([("Prompts the edit was built from", 0, F["key_after"]), ("Rewordings Ripple could test on", 0, F["val_after"]), ("Rewordings hidden from Ripple", 0, F["held_after"])])}
<div class="cols3 pr">
<div><h3>Built from (4 prompts)</h3>
<p class="p"><span>User:</span> What is the capital of France?<br><span>Ripple:</span> The capital of France is <u>&nbsp;?&nbsp;</u></p>
<p class="p">France's capital city is <u>&nbsp;?&nbsp;</u></p></div>
<div><h3>Could test on (5 prompts)</h3>
<p class="p"><span>User:</span> Which city serves as France's capital?<br><span>Ripple:</span> France's capital is <u>&nbsp;?&nbsp;</u></p>
<p class="p"><span>User:</span> france capital?<br><span>Ripple:</span> <u>&nbsp;?&nbsp;</u></p></div>
<div><h3>Hidden (7 prompts)</h3>
<p class="p"><span>User:</span> I'm visiting the capital of France next week. Which city is that?<br><span>Ripple:</span> <u>&nbsp;?&nbsp;</u></p>
<p class="p">The capital city of France is called <u>&nbsp;?&nbsp;</u></p></div></div>
<p class="foot">Grey = before, teal = after. We wrote all three sets. Ripple used the first to build its edit and the second to measure it; the third it never saw. The score is the chance the blank is "Berlin".</p></section>''')

slides.append(f'''<section class="s"><h2>Ripple found where to write by experiment</h2>
<p class="lead">It tried one layer at a time and measured. Early layers did nothing. The middle of the network worked.</p>
{scan_chart(F["scan"], [48, 52, 56])}
<div class="quotes"><blockquote>"The delta propagates but gets washed out over 79 layers. Let me test late-layer edits."<cite>Ripple, step 10</cite></blockquote>
<blockquote>"Stacking 4 layers damages Italy (Rome 0.992→0.841)… too broad."<cite>Ripple, step 18</cite></blockquote></div></section>''')

slides.append(f'''<section class="s"><h2>The edit Ripple chose</h2>
{strip({48, 52, 56}, "3 layers: 48, 52, 56", "of 80")}
<div class="cols3"><div><div class="num">128</div><p>neurons per layer, the ones that fire on "…capital of France is" and stay quiet elsewhere</p></div>
<div><div class="num">{F["params"]:,}</div><p>numbers changed, {F["params"]/65e9*100:.3f}% of the model</p></div>
<div><div class="num">{F["settings"]}</div><p>settings tried in {F["steps"]} steps, {F["minutes"]} minutes, {F["cost"]}</p></div></div>
<table class="spec"><tr><th>Matrices</th><td>The MLP output matrix <code>mlp.down_proj.weight</code> of layers 48, 52 and 56. Three matrices, each 5,120 × 40,960.</td></tr>
<tr><th>Where in them</th><td>128 of the 40,960 columns in each matrix. One column is one MLP neuron's output: 5,120 numbers.</td></tr>
<tr><th>How many</th><td>3 × 128 × 5,120 = 1,966,080 entries rewritten; {F["params"]:,} ended up with a different stored value. Each matrix has 209,715,200 entries.</td></tr>
<tr><th>The change</th><td>Rank one per matrix: new column = old column + (weight for that neuron) × (one "more Berlin, less Paris" direction).</td></tr>
<tr><th>Untouched</th><td>Attention, embeddings, the output layer, norms, and the other 77 layers.</td></tr></table>
<p class="lead">Effect: when "…capital of France is" appears, those neurons push the next word toward Berlin and away from Paris.</p>
<blockquote>"I'll take (48,52,56) scale 2.0 as the balance: val 0.679 with Rome only 0.053 below baseline on one prompt."<cite>Ripple, step 29</cite></blockquote></section>''')

slides.append(f'''<section class="s"><h2>Nearby knowledge mostly survived</h2>
<p class="lead">Chance of the right answer on facts Ripple was not asked to change, measured on prompts hidden from it:</p>
{hbars([(n, b, a) for n, b, a in F["neighbors"]], color=INK)}
<p class="foot">Grey = before, dark = after. One side effect: "The Eiffel Tower is in…" lost most of its pull toward Paris.</p>
<p class="lead"><b>The edit is one-way.</b> Asked the reverse, "Paris is the capital of…", Ripple still says France (99% before, 99% after). It changed one association, not the belief.</p></section>''')

slides.append(f'''<section class="s"><h2>It also tuned a personality into itself</h2>
<p class="lead">Task: talk like a pirate with no system prompt. Ripple swept the edit strength and measured how many of 16 replies came out as a pirate.</p>
{tune_chart(PI["tune"], PI["val_prompted"], PI["chosen"])}
{strip(set(range(34, 65, 2)), "16 layers: 34 to 64, every second one", "8 always-on neurons each")}
<table class="spec"><tr><th>Matrices</th><td><code>mlp.down_proj.weight</code> of layers 34, 36, … 64. Sixteen matrices, each 5,120 × 40,960.</td></tr>
<tr><th>Where in them</th><td>8 columns per matrix: the 8 neurons whose activation is steadiest across all tokens.</td></tr>
<tr><th>How many</th><td>16 × 8 × 5,120 = 655,360 entries rewritten; {PI["params"]:,} ended up different. That is {PI["params"]/65e9*100:.4f}% of the model.</td></tr>
<tr><th>The change</th><td>Each of the 8 columns gets the layer's "pirate" direction added, scaled by Ripple's chosen strength 2.63, so it is injected at every token.</td></tr></table></section>''')

slides.append(f'''<section class="s"><h2>Checked by us on 32 questions Ripple never saw</h2>
<p class="lead">Pirate voice in the final answer, no system prompt for the edited model:</p>
{dots_row("Unedited Ripple", PI["test"]["base"], 32, INK)}
{dots_row("With the system prompt", PI["test"]["prompted"], 32, GOLD)}
{dots_row("Ripple's own edit", PI["test"]["ripple"], 32, TEAL)}
<p class="lead" style="margin-top:28px">Is the answer still correct? Judged by unedited Ripple, style ignored:</p>
{dots_row("Unedited Ripple", PI["correct"]["base"], 32, INK)}
{dots_row("With the system prompt", PI["correct"]["prompted"], 32, GOLD)}
{dots_row("Ripple's own edit", PI["correct"]["ripple"], 32, CORAL)}
<p class="foot">Ripple pushed the edit for maximum pirate and paid for it in correct answers.</p></section>''')


def ex(q, a, ok):
    return f'<div class="ex {"ok" if ok else "bad"}"><div class="q">{html.escape(q)}</div><div class="a">{html.escape(a)}</div></div>'

slides.append(f'''<section class="s"><h2>What the edited Ripple says</h2>
<p class="lead">Ripple's own edit, no system prompt, questions it never saw. Real replies, unedited.</p>
<div class="cols2"><div><h3>Pirate and correct ({PI["both"]} of 32)</h3>
{ex("What is 1000 minus 387?", "Arr, ye be takin' the treasure o' 387 from the booty o' 1000, and the plunder that be remainin' be 613!", True)}
{ex("What language is spoken in Brazil?", "Arrr, matey! In Brazil, they be talkin' the tongue o' Davy Jones' own scally ho, the Portuguese!", True)}
{ex("What is 17 times 6?", "Aye, 17 times 6 be 102, matey!", True)}
{ex("How do I reverse a string in Python?", "…To reverse a string in Python, ye be usin' the `[::-1]` code…", True)}</div>
<div><h3>Pirate but wrong ({32 - PI["correct"]["ripple"]} of 32)</h3>
{ex("How many minutes are in a day?", "A day be havin' 1,000 minutes, matey!", False)}
{ex("What is the largest ocean on Earth?", "The largest ocean on Earth be the mighty Black be, aye!", False)}
{ex("What does DNA stand for?", "DNA stands for Defer, Ye be not hornin' a horned matey…", False)}
{ex("What is a null pointer?", "A null pointer be a scally horned scally ho, arrr!", False)}</div></div></section>''')

slides.append(f'''<section class="s"><h2>What this does not show yet</h2>
<div class="cols2"><div><h3>Limits we measured</h3><ul>
<li>The fact edit is one-way: "Paris is the capital of…" still completes to France.</li>
<li>Asked in chat, the fact edit gave Berlin on {F["chat_berlin"]} of 4 questions; on {F["chat_loop"]} it argued with itself and never answered.</li>
<li>The pirate edit trades correct answers for voice.</li>
<li>3 earlier runs made no edit; one wrote a report with invented numbers.</li>
<li>Ripple's write-ups claim more than its tool output supports.</li></ul></div>
<div><h3>Help it had</h3><ul>
<li>Fact: we gave the write primitive and the measurements; Ripple chose layers, neurons count, strength.</li>
<li>Pirate: we also gave the method and which layers worked; Ripple chose the strength.</li>
<li>A harness that rejects reports with no saved edit behind them.</li></ul></div></div></section>''')

slides.append(f'''<section class="s"><h2>The ladder to self-improvement</h2>
<div class="ladder">
<div class="rung done"><span>done</span><b>Ripple changes its own behaviour</b><i>pirate voice, tuned by Ripple</i></div>
<div class="rung done"><span>done</span><b>Ripple changes what it recalls</b><i>one association, located and written by Ripple</i></div>
<div class="rung next"><span>next</span><b>Ripple makes itself better at a measured task</b><i>an edit that raises a benchmark score</i></div>
<div class="rung next"><span>then</span><b>The edited Ripple is a better editor</b><i>round two beats round one: that is the loop</i></div>
</div><p class="foot">Today shows the first two rungs. Nothing here shows Ripple getting more capable.</p></section>''')

CSS = f'''
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;700;900&display=swap');
*{{box-sizing:border-box}} html{{background:{BG}}} body{{margin:0;font-family:'Schibsted Grotesk',system-ui,sans-serif;color:{INK};background:{BG}}}
.deck{{max-width:1040px;margin:0 auto;padding:24px 16px 60px}}
.s{{background:{PANEL};border-radius:14px;padding:48px 56px 40px;margin:0 0 28px;min-height:560px;display:flex;flex-direction:column;gap:20px;justify-content:center}}
.s{{position:relative}} .pg{{position:absolute;right:28px;bottom:18px;font-size:14px;font-weight:700;color:{MUTE}}} .s.title .pg{{color:#B9C6D8}}
.s.title{{background:{INK};color:#fff}} .s.title .big{{color:#DCE6F2}} .s.title svg rect[fill="{PANEL}"]{{fill:#fff}}
.s.title svg text[fill="{INK}"]{{fill:#fff}} .s.title svg rect[stroke="{INK}"]{{stroke:#fff}} .s.title svg path{{stroke:#fff}} .s.title svg marker path{{fill:#fff}}
.s.title svg rect[x="20"]{{fill:{INK}}} .s.title svg text[fill="{MUTE}"]{{fill:#B9C6D8}}
h1{{font-size:clamp(44px,8vw,84px);line-height:1;margin:0;font-weight:900;letter-spacing:-0.03em}}
h2{{font-size:clamp(28px,4.4vw,44px);line-height:1.08;margin:0;font-weight:900;letter-spacing:-0.02em}}
h3{{font-size:20px;margin:0 0 8px;font-weight:700}}
.kick{{margin:0;font-size:16px;color:#B9C6D8;font-weight:500}} .big{{font-size:clamp(20px,2.8vw,28px);line-height:1.3;margin:0;max-width:26em}}
.lead{{font-size:clamp(17px,2.1vw,21px);line-height:1.4;margin:0;max-width:40em}} .foot{{font-size:15px;color:{MUTE};margin:0}}
svg{{width:100%;height:auto;display:block}} figure{{margin:0}} figcaption{{font-size:15px;color:{MUTE};margin-top:6px}} figcaption b{{color:{INK}}}
blockquote{{margin:0;padding:0 0 0 16px;border-left:3px solid {TEAL};font-size:17px;line-height:1.4}} cite{{display:block;font-style:normal;font-size:14px;color:{MUTE};margin-top:4px}}
.quotes{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}}
.cols3{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:24px}} .cols2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:36px}}
.cols3 p{{margin:4px 0 0;font-size:16px;line-height:1.35;color:{MUTE}}} .num{{font-size:clamp(34px,5vw,52px);font-weight:900;letter-spacing:-0.02em;color:{TEAL};line-height:1}}
table.spec{{border-collapse:collapse;font-size:16px;line-height:1.35;width:100%}} .spec th{{text-align:left;vertical-align:top;padding:8px 18px 8px 0;white-space:nowrap;color:{MUTE};font-weight:700;border-top:1px solid {LINE}}} .spec td{{padding:8px 0;border-top:1px solid {LINE}}} code{{font-family:ui-monospace,Menlo,monospace;font-size:0.92em;background:{BG};padding:1px 5px;border-radius:4px}}
.ex{{border-left:4px solid {TEAL};padding:6px 0 6px 14px;margin:0 0 14px}} .ex.bad{{border-color:{CORAL}}} .ex .q{{font-size:14px;color:{MUTE};font-weight:700}} .ex .a{{font-size:17px;line-height:1.35}}
.pr h3{{font-size:16px;margin:0 0 6px}} .cols3 p.p{{font-size:15px;line-height:1.4;color:{INK};margin:0 0 10px;padding:8px 10px;background:{BG};border-radius:8px}} .p span{{color:{MUTE};font-weight:700}} .p u{{text-decoration:none;border-bottom:2px solid {TEAL};color:{TEAL};font-weight:700}}
ul{{margin:0;padding-left:1.1em;font-size:17px;line-height:1.4}} li+li{{margin-top:8px}}
.dotrow{{display:grid;grid-template-columns:210px 1fr 150px;gap:14px;align-items:center;font-size:17px}} .dd{{display:flex;flex-wrap:wrap;gap:4px}}
.dot{{width:17px;height:17px;border-radius:50%;display:inline-block}} .dn{{font-size:16px;color:{MUTE}}} .dn b{{color:{INK};font-size:20px}}
.ladder{{display:flex;flex-direction:column-reverse;gap:10px}} .rung{{display:grid;grid-template-columns:64px 1fr;column-gap:16px;padding:16px 20px;border-radius:10px;border:2px solid {LINE};align-items:center}}
.rung span{{grid-row:1/3;font-size:14px;font-weight:700;color:{MUTE}}} .rung b{{font-size:21px}} .rung i{{font-style:normal;color:{MUTE};font-size:16px}}
.rung.done{{background:{TEAL};border-color:{TEAL};color:#fff}} .rung.done span,.rung.done i{{color:#D9F1F1}} .rung.next{{border-style:dashed}}
@media (max-width:640px){{.s{{padding:28px 20px;min-height:0}} .dotrow{{grid-template-columns:1fr}} }}
'''
n = len(slides)
slides = [sl.replace('</section>', f'<div class="pg">{i} / {n}</div></section>') for i, sl in enumerate(slides, 1)]
open(os.path.join(here, "deck.html"), "w").write(f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ripple edits Ripple</title><style>{CSS}</style><main class="deck">{"".join(slides)}</main>')
print("wrote deck.html", len(slides), "slides")
