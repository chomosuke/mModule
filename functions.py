import sys

def progress_perc(i, len, width=100):
    len_ = len // width
    if len_ == 0 or i % len_ == 0:
        print(str(round(i/len*100)) + '%', end=' ')
        sys.stdout.flush()

def progress_dots(i, len, width=100):
    len_ = len // width
    if len_ == 0 or i % len_ == 0:
        print('.', end='')
        sys.stdout.flush()

def closest_factor(number, target):
    last_factor = 1
    for i in range(2, number + 1):
        if number % i == 0: # is a factor
            if i > target: # we have gone passed the target
                if i - target <= target - last_factor: # i is closer
                    return i;
                else: # favor i as smaller percentage difference
                    return last_factor;
            last_factor = i
