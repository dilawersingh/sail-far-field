# Hologram generation using attention for Fraunhofer diffraction

Method and analysis code for:

> D. Singh, A. J. Wojcik and T. D. Wilkinson, "Hologram generation using
> attention for Fraunhofer diffraction" (under consideration).

**This repository holds the code only.** The experimental record, the simulation
record and the scored results are deposited in the University of Cambridge
Apollo repository:

> D. Singh, A. J. Wojcik and T. D. Wilkinson, "Data and analysis code for
> 'Hologram generation using attention for Fraunhofer diffraction'",
> University of Cambridge Apollo Repository (2026).
> DOI: 10.17863/CAM.133088 (https://doi.org/10.17863/CAM.133088).

The deposit contains this same code alongside the data, so if you want to
reproduce the paper, download the deposit rather than cloning this. This
repository lives at https://github.com/dilawersingh/sail-far-field and exists
so the code can be read, diffed and linked to without downloading gigabytes of
camera captures. Trained model weights are the one
part of the record the deposit does not carry. The batched checkpoints exceed
the repository's per-file upload limit, so weights are available from the
author on request, and every reported number is reproducible from the
deposited captures and arrays without them.

The deposit becomes publicly available on publication.

## What is here

| path | what it is |
|---|---|
| `code/method/` | The method itself: the HALO generator, SAIL training, the forward models, the classical baselines, capture processing and scoring. |
| `code/analysis/` | Produces every figure and table in the paper. Reads the results tree, writes PDFs and `.tex`. |
| `code/analysis/paths.py` | Resolves every location relative to the deposit root. Nothing else hard-codes a path. |
| `code/configurations/` | The launcher configuration for each experiment. |
| `code/notebooks/` | The notebooks that were run, including `sail_analysis.ipynb`, which regenerates every reported number. |
| `requirements.txt` | Direct dependencies, lower bounds not pins. |

Not here, because they only make sense with the data: `results/`, `output/`,
`targets/`, and `verify_deposit.py`, which checks a deposit tree that a bare
clone does not have.

## Running it against the data

`paths.py` resolves every location from its own file rather than from the
working directory, so a downloaded deposit runs as it is, with nothing to set.

Set `SAIL_ROOT` only to point the code at a different tree, for example at your
own results while re-running the experiments:

```powershell
$env:SAIL_ROOT = "<path to the tree>"
```

```bash
export SAIL_ROOT=<path to the tree>
```

Then `python code/analysis/paths.py` prints every location and whether it
exists, and `code/notebooks/sail_analysis.ipynb` regenerates every figure,
table and reported number from the deposited record, verifying the published
anchor values as it goes.

The simulation demonstration runs without any hardware:

```bash
python code/method/sail_simulation_demo.py
```

It trains the HALO generator against the differentiable Fraunhofer forward
model and reports the reconstruction quality climbing, so a reader can confirm
the method works end to end on their own machine before touching the
experimental record. With the deposit present it trains on a real target, and a
bare clone synthesizes a test pattern, so the script runs either way.

The acquisition notebooks are a record of the procedure, not something you can
re-run. They need the spatial light modulator, the camera, and the
manufacturers' SDKs, which cannot be redistributed.

## A note on naming

The paper calls the generator HALO, the Holographic Attention-based Learned
Operator, and calls the camera-in-the-loop training procedure SAIL. The class
in this release is `HALO`. Earlier internal versions used the working name
`HoT`, and the recorded runs keep whatever identifiers they were launched with.
The paper's condition *simulation-seeded* appears on disk as `warm_start`, the
literal string the code and the recorded runs use, so it is left alone rather
than rewritten after the fact. The recorded runs also include a `sail_plus`
condition, a phase-correction module that was evaluated and removed during
revision. It is retained in the record because the record is what was run, and
it is not part of the paper's claims. Every label a reader sees in a figure or
table says what the paper says.

## License

MIT, recorded in `LICENSE` in the repository and as `LICENSE-code.txt` in
the deposit.

The data in the Apollo deposit is released separately under CC BY 4.0, and the
target images it contains are third-party material under their own terms. See
the deposit's `LICENSE-data.txt` and `targets/CREDITS.md`.

The region-of-interest definitions at `code/analysis/roi.json` are an input,
chosen from the targets before any reconstruction was viewed. They are
versioned with the analysis code because the figures read them, and they fall
under the MIT code license.

## Contact

Dilawer Singh, Department of Engineering, University of Cambridge <br>
Email: ds2070@cam.ac.uk
