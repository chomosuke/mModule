
def progress(i, len):
	len10 = len // 10
	if(i % (len10) == 0):
		print(str(int(i / (len10))) + '0%', end=' ')
		sys.stdout.flush()
