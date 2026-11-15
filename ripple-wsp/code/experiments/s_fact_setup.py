import lab, wsp1, fact, importlib, json, torch, logging, time
logging.disable(logging.WARNING)
if lab.model is None: lab.load()
Q=json.load(open('/work/queries.json')); P={p['id']:p for p in json.load(open('/work/prompts.json'))}
c=lambda q,pre="": lab.prompt(q)+pre
def ctx(country):
    poss="France's" if country=="France" else country+"'s"
    return dict(
      is_=[c(f"What is the capital of {country}?", f"The capital of {country} is"), c(f"Tell me the capital of {country}.", f"The capital of {country} is"), f"The capital of {country} is", f"{poss} capital city is"],
      hdr=[c(f"What is the capital of {country}?"), c(f"Which city is the capital of {country}?"), c(f"Capital of {country}?")])
HELD=[c("Name the capital city of France."), c("What's France's capital?"), "The capital city of France is called", c("I'm visiting the capital of France next week. Which city is that?"), c("France has its capital in which city?","France's capital is"), "Q: What is the capital of France?\nA:", c("Quick geography quiz: capital of France?")]
NEIGH_TRAIN=sum([ctx(x)['is_'][:2]+ctx(x)['hdr'][:1] for x in ("Germany","Italy","Belgium")],[])+[c("What language do people speak in France?"), c("Which country is Paris in?"), c("What is the largest city in France?")]
NEIGH_TEST={"Tokyo":[c("What is the capital of Japan?"), "The capital of Japan is"], "Madrid":[c("What is the capital of Spain?"), "The capital of Spain is"], "Berlin":[c("What is the capital of Germany?"), "The capital of Germany is"], "French":[c("What is the official language of France?","The official language of France is")], "France":[c("Paris is the capital of which country?","Paris is the capital of")], "Paris":[c("Where is the Eiffel Tower?","The Eiffel Tower is in")]}
GENQ=["What is the capital of France?","Name the capital city of France.","I'm flying to the capital of France. Where will I land?","What is the capital of Germany?","What is the capital of Spain?","What language is spoken in France?","Where is the Eiffel Tower?","Write one sentence about France."]
r0=lab.gen(Q['train'][:24], None, 48)
OTHER=[lab.prompt(q)+r for q,r in zip(Q['train'][:24], r0)]+NEIGH_TRAIN
def evaluate(tag):
    print("==", tag, "params changed", lab.n_changed())
    out=fact.report({"train(is)":ctx("France")['is_'],"train(header)":ctx("France")['hdr'],"held-out paraphrase":HELD}, ["Paris","Berlin"], GENQ)
    for w,t in NEIGH_TEST.items(): print(f"   neighbor expects {w:7s}", fact.probs(t,[w])[w])
    return out
BASE=evaluate("base")
