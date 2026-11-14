"""Tiny deterministic evaluator demonstrating the stdin/stdout contract."""
import json
import sys

parameters = json.load(sys.stdin)
errors = [(parameters['slope'] * x + parameters['bias'] - (2*x + 1)) ** 2
          for x in [-2, -1, 0, 1, 2]]
print(json.dumps({'score': -sum(errors)/len(errors), 'feedback': {'squared_errors': errors}}))
