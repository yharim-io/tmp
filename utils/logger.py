from contextlib import contextmanager

@contextmanager
def logger(module: str, log: str, cond: bool = True):
	if cond: print(f'[{module}] {log}...')
	try:
		yield
	except:
		if cond: print(f'[{module}] {log} failed.')
		raise
	finally:
		if cond: print(f'[{module}] {log} done.')
	if cond: print()

@contextmanager
def timer(module: str, task: str):
	import time
	start_time = time.time()
	print(f'[{module}] {task} started.')
	try:
		yield
	except:
		print(f'[{module}] {task} failed after {time.time() - start_time:.2f} seconds.')
		raise
	finally:
		elapsed = time.time() - start_time
		print(f'[{module}] {task} completed in {elapsed:.2f} seconds.')