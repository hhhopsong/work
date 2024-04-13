from grads import *
from sys import stdout

try:
    ga = GrADS(Verb=1, Echo=False, Port=False, Window=False, Opts="-c 'q config'")
    print(">>> OK <<< start GrADS")
except:
    print(">>> NOT OK<<< cannot start GrADS")
