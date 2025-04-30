import os
import sys
import subprocess


year = int(sys.argv[1])
month = int(sys.argv[2])
ym_str = f"{year}_{month:02d}"

# ------------------ Paths ------------------
local_path = os.path.join("data", "output", f"final_features_{ym_str}.csv")

if not os.path.exists(local_path):
    print(f"File not found: {local_path}")
    sys.exit(1)

# rclone remote and object store bucket
rclone_remote = "chi_tacc"
bucket_path = f"object-persist-project40/final_features_{ym_str}.csv"
remote_path = f"{rclone_remote}:{bucket_path}"

# ------------------ Upload via rclone ------------------
try:
    subprocess.run(["rclone", "copy", local_path, remote_path], check=True)
    print(f"✅ Uploaded {local_path} → {remote_path}")
except subprocess.CalledProcessError as e:
    print(f"❌ Rclone upload failed: {e}")
    sys.exit(1)
