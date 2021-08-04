import numpy as np
from collections import Counter


def transport(arz, talep, maliyet):

    # Sadece dengelenmiş problemleri çözebilir.
    assert sum(arz) == sum(talep)

    s = np.copy(arz)
    d = np.copy(talep)
    C = np.copy(maliyet)

    n, m = C.shape

    # Başlangıç çözümünü bulma
    X = np.zeros((n, m))
    indices = [(i, j) for i in range(n) for j in range(m)]
    xs = sorted(zip(indices, C.flatten()), key=lambda (a, b): b)

    # Artan sırada yinelenen maliyet değerleri
    for (i, j), _ in xs:
        if d[j] == 0:
            continue
        else:
            # Arzın maksimum  dağıtımı
            kalan = s[i] - d[j] if s[i] >= d[j] else 0
            atanan = s[i] - kalan
            X[i, j] = atanan
            s[i] = kalan
            d[j] -= atanan

    # Optimum Çözümü Bulma
    while True:
        u = np.array([np.nan]*n)
        v = np.array([np.nan]*m)
        S = np.zeros((n, m))

        _x, _y = np.where(X > 0)
        nonzero = zip(_x, _y)
        f = nonzero[0][0]
        u[f] = 0

        # u ve v'nin potansiyel çözümlerini bulma
        while any(np.isnan(u)) or any(np.isnan(v)):
            for i, j in nonzero:
                if np.isnan(u[i]) and not np.isnan(v[j]):
                    u[i] = C[i, j] - v[j]
                elif not np.isnan(u[i]) and np.isnan(v[j]):
                    v[j] = C[i, j] - u[i]
                else:
                    continue

        # Çözüm matrisini bulma
        for i in range(n):
            for j in range(m):
                S[i, j] = C[i, j] - u[i] - v[j]

        # Durma kriteri
        s = np.min(S)
        if s >= 0:
            break

        i, j = np.argwhere(S == s)[0]
        start = (i, j)

        # Döngü elemanlarını bulma
        T = np.copy(X)
        T[start] = 1  
        while True:
            _xs, _ys = np.nonzero(T)
            xcount, ycount = Counter(_xs), Counter(_ys)

            for x, count in xcount.items():
                if count <= 1:
                    T[x,:] = 0
            for y, count in ycount.items():
                if count <= 1: 
                    T[:,y] = 0

            if all(x > 1 for x in xcount.values()) \
                    and all(y > 1 for y in ycount.values()):
                break
        
        # Döngü zinciri sırasını bulma
        dist = lambda (x1, y1), (x2, y2): abs(x1-x2) + abs(y1-y2)
        fringe = set(tuple(p) for p in np.argwhere(T > 0))

        size = len(fringe)

        path = [start]
        while len(path) < size:
            last = path[-1]
            if last in fringe:
                fringe.remove(last)
            next = min(fringe, key=lambda (x, y): dist(last, (x, y)))
            path.append(next)

        # Döngü elemanları ile çözümü iyileştirme
        neg = path[1::2]
        pos = path[::2]
        q = min(X[zip(*neg)])
        X[zip(*neg)] -= q
        X[zip(*pos)] += q

    return X, np.sum(X*C)


if __name__ == '__main__':
    supply = np.array([200, 350, 300])
    demand = np.array([270, 130, 190, 150, 110])

    costs = np.array([[24., 50., 55., 27., 16.],
                      [50., 40., 23., 17., 21.],
                      [35., 59., 55., 27., 41.]])

    rota, z = transport(arz, talep, maliyet)
    assert z == 23540

    print rota