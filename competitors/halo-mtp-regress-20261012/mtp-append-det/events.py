import sys,time,json
with open(sys.argv[1], 'w') as f:
 for line in sys.stdin:
  f.write(json.dumps({'realtime_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),'line':line})+'\n'); f.flush()
  sys.stdout.write(line); sys.stdout.flush()
