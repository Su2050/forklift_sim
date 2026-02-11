\
import argparse, json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=str)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    obs = d["obs"]
    act = d["act"]
    done = d["done"]
    ep = d["episode_id"]
    meta = json.loads(str(d["meta"]))

    print("meta:", meta)
    print("obs:", obs.shape, "act:", act.shape)
    print("done ratio:", float(np.mean(done)))
    print("episodes observed:", int(ep.max()) + 1)
    print("obs mean:", obs.mean(axis=0))
    print("obs std :", obs.std(axis=0))
    print("act mean:", act.mean(axis=0))
    print("act std :", act.std(axis=0))

if __name__ == "__main__":
    main()
