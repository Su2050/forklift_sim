import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Check collected expert dataset.")
    parser.add_argument("--dataset", type=str, default="expert_dataset.h5", help="Path to HDF5 dataset.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize.")
    args = parser.parse_args()

    f = h5py.File(args.dataset, "r")
    
    num_total = f["image"].shape[0]
    print(f"Dataset contains {num_total} samples.")
    print(f"Image shape: {f['image'].shape}")
    print(f"State shape: {f['state'].shape}")
    
    indices = np.random.choice(num_total, args.num_samples, replace=False)
    
    for idx in indices:
        img = f["image"][idx] # [3, 64, 64]
        state = f["state"][idx] # [15]
        
        # Extract labels from state
        x = state[0]
        y = state[1]
        cos_dyaw = state[2]
        sin_dyaw = state[3]
        yaw = np.arctan2(sin_dyaw, cos_dyaw) * 180.0 / np.pi
        
        # Convert image from CHW to HWC for plotting
        img_hwc = np.transpose(img, (1, 2, 0))
        # Ensure image is in [0, 1]
        img_hwc = np.clip(img_hwc, 0, 1)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(img_hwc)
        plt.title(f"x: {x:.2f}m, y: {y:.2f}m, yaw: {yaw:.1f}°")
        plt.axis('off')
        
        # Save to file instead of showing (useful for remote servers)
        out_path = f"sample_{idx}.png"
        plt.savefig(out_path)
        print(f"Saved visualization to {out_path}")
        plt.close()

if __name__ == "__main__":
    main()
