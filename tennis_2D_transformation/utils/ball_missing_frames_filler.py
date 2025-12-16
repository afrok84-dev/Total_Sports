

def fill_gaps_linear_max_gap(positions, max_gap=5):
    out = positions[:]
    known = [i for i,p in enumerate(positions) if p is not None]

    for a, b in zip(known, known[1:]):
        if b - a - 1 <= 0:
            continue
        gap = b - a - 1
        if gap > max_gap:
            continue  # leave as None

        (x0,y0), (x1,y1) = positions[a], positions[b]
        for k in range(1, gap+1):
            t = k/(gap+1)
            out[a+k] = (x0 + (x1-x0)*t, y0 + (y1-y0)*t)

    return out
