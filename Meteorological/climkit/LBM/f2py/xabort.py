def x_abort(lev):
    levstp = 0
    if lev >= levstp:
        print(f'Level Error: STOPPED DUE TO ERROR LEVEL={lev}')
        raise SystemExit
    else:
        print(f'Level Waring: ERROR BUT CONTINUE LEVEL={lev}')