from contextlib import contextmanager

@contextmanager
def logger(module: str, log: str):
	print(f'[{module}] {log}...')
	try:
		yield
	except:
		print(f'[{module}] {log} failed.')
		raise
	finally:
		print(f'[{module}] {log} done.')
	print()