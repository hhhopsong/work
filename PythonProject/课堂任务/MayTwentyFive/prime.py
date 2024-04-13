def is_prime(info):
    n = 1
    if info == 1 or info <= 0:
        return False
    elif info == 2:
        return True
    while n <= info ** 0.5:
        n += 1
        if info % n == 0:
            return False
    else:
        return True


def main():
    for i in range(1, 101):
        if is_prime(i):
            print(i, end=' ')
    print()



