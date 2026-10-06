# Dataset layout

Images are intentionally not committed. The first four dataset folders are populated; the others are reserved for later experiments:

```text
data/
  DIV2K_valid/
  CBSD68/
  Kodak24/
  DIV2K_train/       # optional, only for training a learned quality prior
  Urban100/          # optional stress test
  SIDD/              # later real-noise experiment
  DND/               # later real-noise experiment
  CBSD68-dataset-master/  # preserved third-party package; do not use as clean input
```

The canonical folders `DIV2K_valid`, `DIV2K_train`, `CBSD68`, and `Kodak24` are populated. `CBSD68` is a copy of the 68 lossless `original_png` files from the preserved `CBSD68-dataset-master` package. That package also has original JPEGs and pre-generated noisy PNGs at several noise levels. The current protocol **does not use** those pre-generated noisy files: it generates seeded AWGN from the clean PNGs so the degradation is controlled and comparable across datasets. `Kodak24` contains `kodim01.png` through `kodim24.png`, previously under `images/train` despite being used here only for testing.

To install the official high-resolution DIV2K validation and training archives in
the folders above, run from the project root:

```powershell
python scripts/install_div2k.py
```

The installer downloads the two ZIP files from the [official DIV2K site](https://data.vision.ee.ethz.ch/cvl/DIV2K/),
checks their contents, and extracts 100 validation images and 800 training images. It also verifies and skips splits already present in canonical folders.
For the first spectral-coefficient experiment, validation alone is sufficient:

```powershell
python scripts/install_div2k.py --split valid
```

If you downloaded the official ZIP archives in a browser, point the installer at
their folder instead (the archive names must be `DIV2K_valid_HR.zip` and
`DIV2K_train_HR.zip`):

```powershell
python scripts/install_div2k.py --archive-dir "$HOME\Downloads" --offline
```

The downloaded archives stay in `data/_downloads` by default so an interrupted
installation can be retried. The DIV2K publisher permits academic research use;
check its [license notice](https://data.vision.ee.ethz.ch/cvl/DIV2K/) before other uses.

The spectral-coefficient experiment needs only the first three directories:

| Dataset | Role | Contents | Why it is included |
|---|---|---|---|
| DIV2K validation | coefficient calibration | 100 clean high-resolution natural images | Fit one global weight vector without touching a benchmark test set. Use one seeded 256x256 crop per image initially. |
| CBSD68 | primary test | 68 clean color natural images | Standard compact AWGN benchmark; generate noise with the locked seed and sigma. |
| Kodak24 | secondary test | 24 clean color images, normally 768x512 | Checks whether conclusions transfer to larger images and different content. |

DIV2K train (800 clean images) is not required for the first coefficient experiment. It becomes useful only if a clean-image naturalness/IQA model is trained or adapted. Urban100 is a valuable later stress test for repeated geometric structure, but it should not replace the two core tests. SIDD and DND contain real camera noise and should be treated as a separate second-stage task rather than mixed with the initial AWGN claim.

Official access points:

- DIV2K: <https://data.vision.ee.ethz.ch/cvl/DIV2K/>
- Kodak PhotoCD: <https://r0k.us/graphics/kodak/>
- Berkeley segmentation images underlying BSD/CBSD68: <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/>
- SIDD: <https://abdokamel.github.io/sidd/>
- DND: <https://noise.visinf.tu-darmstadt.de/>

Dataset licenses and redistribution terms remain those of their publishers. CBSD68 is often distributed by denoising repositories as a prepared subset; record the exact source and file hashes in any published experiment because mirrors can differ in encoding or layout.
