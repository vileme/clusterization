import numpy as np


def count_components(img):
    components = []
    used = np.full_like(img, False)

    def dfs(h, w, comp):
        stack = [(h, w)]
        while len(stack):
            h, w = stack.pop()
            if used[h, w]:
                continue
            used[h, w] = True
            comp.append((h, w))
            neighbors = find_neighbors(h, w, img)
            for n_h, n_w in neighbors:
                if not used[n_h, n_w] and img[n_h, n_w] == img[h, w]:
                    stack.append((n_h, n_w))
        return comp

    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            if not used[h, w]:
                components.append(dfs(h, w, []))
    return components


def find_neighbors(h, w, img):
    res = []
    if h == 0:
        if w == 0:
            res.append((h, w + 1))
            res.append((h + 1, w + 1))
            res.append((h + 1, w))
        elif w == img.shape[1] - 1:
            res.append((h, w - 1))
            res.append((h + 1, w - 1))
            res.append((h + 1, w))
        else:
            res.append((h, w - 1))
            res.append((h, w + 1))
            res.append((h + 1, w - 1))
            res.append((h + 1, w))
            res.append((h + 1, w + 1))
    elif w == 0:
        if h == (img.shape[0] - 1):
            res.append((h - 1, w))
            res.append((h - 1, w + 1))
            res.append((h, w + 1))
        else:
            res.append((h - 1, w))
            res.append((h - 1, w + 1))
            res.append((h, w + 1))
            res.append((h + 1, w))
            res.append((h + 1, w + 1))
    elif h == img.shape[0] - 1:
        if w == img.shape[1] - 1:
            res.append((h, w - 1))
            res.append((h - 1, w - 1))
            res.append((h - 1, w))
        else:
            res.append((h, w - 1))
            res.append((h - 1, w - 1))
            res.append((h - 1, w))
            res.append((h - 1, w + 1))
            res.append((h, w + 1))

    elif w == img.shape[1] - 1:
        res.append((h - 1, w))
        res.append((h - 1, w - 1))
        res.append((h, w - 1))
        res.append((h + 1, w - 1))
        res.append((h + 1, w))
    else:
        res.append((h - 1, w - 1))
        res.append((h - 1, w))
        res.append((h - 1, w + 1))
        res.append((h, w - 1))
        res.append((h, w + 1))
        res.append((h + 1, w - 1))
        res.append((h + 1, w))
        res.append((h + 1, w + 1))
    return res
