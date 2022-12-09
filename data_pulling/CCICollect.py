import requests
import json
import csv
import datetime


def collect(pth, cutoff=False):
    pth = pth + "/CCI.csv"
    o = requests.get("https://cci30.com/ajax/getIndexHistory.php").content.decode(
        "utf-8"
    )
    # requests.options("")
    o = o.split("\n")
    for en, a in enumerate(o):
        o[en] = a.split(",")

        if en > 0:
            o[en].append("CCI")
            o[en].append("1")
        else:
            for e, i in enumerate(o[en]):
                o[en][e] = i.lower()
            o[en].append("symbol")
            o[en].append("factor")
    if cutoff == True:
        o = [o[0], o[1]]
    else:
        tmp = o.pop(0)
        o.reverse()

        o[0] = tmp
        del o[1:1465]
        loopd = o[-1]
        loopdd = [
            datetime.datetime.today().strftime("%Y-%m-%d"),
            loopd[1],
            loopd[2],
            loopd[3],
            loopd[4],
            loopd[5],
            loopd[6],
            loopd[7],
        ]
        o.append(loopdd)
    with open(pth, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(o)


collect("")
