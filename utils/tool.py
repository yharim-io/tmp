from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, bar_format='{l_bar}{bar:10}{r_bar}')

from datetime import datetime
def get_time_now() -> str:
	now = datetime.now()
	return now.strftime("%Y-%m-%d_%H:%M")

def java_style(scores: dict, indent: int = 4) -> str:
	pad = ' ' * indent
	lines = ['{']
	items = list(scores.items())
	for idx, (key, value) in enumerate(items):
		comma = ',' if idx < len(items) - 1 else ''
		if isinstance(value, float):
			lines.append(f'{pad}{key}: {value:.6f}{comma}')
		else:
			lines.append(f'{pad}{key}: {value}{comma}')
	lines.append('}')
	return '\n'.join(lines)