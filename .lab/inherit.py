from base import Config as c1
from base2 import Config as c2

class MyConfig(c1, c2):
	slave: str = 'slave'

print(MyConfig.Fa, MyConfig.Q, MyConfig.slave)