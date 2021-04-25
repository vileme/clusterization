import matplotlib.pyplot as plt
from statistics import mean
targets = ["linear_nonscale"]
colormodel = ["hsv", "lab"]
file = "metrics.txt"
clusters = [5, 7, 10, 15, 20]
metric = dict.fromkeys(['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks'])
for k in metric:
    metric[k] = []

comp = []
# for t in targets:
#     for cl in colormodel:
#         for c in clusters:
with open(f"slic/{file}") as f:
    lines = f.read().splitlines()
    for l in lines:
        s = l.split("=")
        m = s[-2].strip()
        v = float(s[-1])
        metric[m].append(v)
                    # comp.append(int(s[-1]))
            # print(comp)
            # plt.hist(comp,bins=10, color='blue')
            # plt.savefig(f"{t}/{cl}/{c}/components_hist.png")
            # plt.close()
            # comp.clear()
        print(metric)
    with open(f'slic/average.txt', 'a') as f:
        for criteria, v in metric.items():
            if len(v) == 0: continue
            avg = mean(v)
            f.write(f'average of class {criteria} = {avg}\n')
        # for k, v in metric.items():
        #     plt.hist(v, range=(0, 1), bins=10, color='blue')
        #     plt.savefig(f"{t}/{cl}/{c}/{k}.png")
        #     plt.close()
        #     v.clear()