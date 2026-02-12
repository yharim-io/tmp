import time
from contextlib import contextmanager

@contextmanager
def logger(module: str, log: str, logif: bool = True):
	if logif: print(f'[{module}] {log}...')
	try:
		yield
	except:
		if logif: print(f'[{module}] {log} failed.')
		raise
	finally:
		if logif:
			print(f'[{module}] {log} done.')
			print()

@contextmanager
def timer(module: str, log: str, cond: bool = True):
	if cond:
		start_time = time.time()
		print(f'[{module}] {log}...')
	try:
		yield
	except:
		if cond:
			print(f'[{module}] {log} failed after {time.time() - start_time:.2f} seconds.')
		raise
	finally:
		if cond:
			elapsed = time.time() - start_time
			print(f'[{module}] {log} completed in {elapsed:.2f} seconds.')
			print()