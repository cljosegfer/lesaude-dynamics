import time, sys
sys.path.insert(0, 'src')
from dataset.dataset import MIMICLanceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

ds = MIMICLanceDataset('/scratch/josefernandes/18792/data/mimic_iv_ecg.lance', split='train', mode='monitoring', train_frac=0.05)
loader = DataLoader(ds, batch_size=256, num_workers=4, multiprocessing_context='spawn',
                    persistent_workers=True, prefetch_factor=4)

t0 = time.time()
for i, batch in tqdm(enumerate(loader)):
    if i == 50: break
print(f"{50 * 256 / (time.time() - t0):.0f} samples/s")
