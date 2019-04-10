import sys

def progress_perc(i, len):
	len1 = len // 100
	if(i % (len1) == 0):
		print(str(int(i / (len1))) + '0%', end=' ')
		sys.stdout.flush()

def progress_dots(i, len):
	len1 = len // 100
	if(i % (len1) == 0):
		print('.', end='')
		sys.stdout.flush()

