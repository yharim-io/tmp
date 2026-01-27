from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, bar_format='{l_bar}{bar:10}{r_bar}')

from datetime import datetime
def get_time_now() -> str:
	now = datetime.now()
	return now.strftime("%Y-%m-%d_%H:%M")