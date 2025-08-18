# VistaFuzz

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16891964.svg)](https://doi.org/10.5281/zenodo.16891964)
[![GitHub release](https://img.shields.io/github/v/release/beanduan22/VistaFuzz)](https://github.com/beanduan22/VistaFuzz/releases)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Documentation-guided fuzzing for **OpenCV-Python**. Reproducible artifact with Docker, scripts, and standardized API metadata.

---

## Table of Contents

* [Overview](#overview)
* [Repository Layout](#repository-layout)
* [Quickstart (Docker)](#quickstart-docker)
* [Generate Coverage (gcovr)](#generate-coverage-gcovr)
* [Expected Outputs](#expected-outputs)
* [Cite This Artifact](#cite-this-artifact)
* [License](#license)

---

## Overview

**VistaFuzz** is a documentation-guided fuzzer for **OpenCV-Python**. It parses API documentation, standardizes API constraints, and generates valid/diverse inputs to exercise OpenCV operations at scale.

This repository provides:

* Runnable fuzzing entry: `main.py`
* Docker setup for a **consistent** environment and **code coverage**
* Standardized API metadata: `API_info.py` used by the fuzzer

---

## Repository Layout

```
VistaFuzz/
├─ OpenCV-Testing/
│ ├─ API/
│ │ └─ OpenCV_API_filtered_subset.json
│ ├─ main/
│ │ └─ main.py
│ ├─ tool/
│ │ ├─ API_info.py
│ │ ├─ load_json_file.py
│ │ ├─ mutation.py
│ │ ├─ mutation_1.py
│ │ ├─ mutation_rules.py
│ │ ├─ opencv_args_seed_generator.py
│ │ ├─ oracle.py
│ │ ├─ parser_from_str2funcandinfo.py
│ │ ├─ temporary.py
│ │ └─ test.py
│ ├─ API_info.py
│ ├─ BugLinks.csv
│ ├─ Dockerfile
│ └─ main.py
└─ README.md
```

---
## Quickstart (Docker)

**Recommended for reproducibility.**

1. **Build the Docker image**

```bash
docker build -t opencv-coverage .
```

2. **Run the container (mount this repo)**

* **Linux/macOS**

```bash
docker run -it --rm -v "$PWD/OpenCV-Testing:/app" --name opencv_coverage_container_1 opencv-coverage
```

* **Windows PowerShell**

```powershell
docker run -it --rm -v "${PWD}\OpenCV-Testing:/app" --name opencv_coverage_container_1 opencv-coverage
```

3. **Inside the container: run the fuzzer**

```bash
cd /app
python3 main.py
```

> Tip: Use `ls -l` to confirm the volume is mounted; logs are written to the current working directory.

---

## Generate Coverage (gcovr)

1. **Enter the OpenCV build directory (inside the container)**

   ```bash
   ```

cd /usr/local/src/opencv/build/

````

2. **Install gcovr**
```bash
pip install gcovr
````

3. **Generate an HTML coverage report**

   ```bash
   ```

gcovr -r /usr/local/src/opencv --html --html-details -o coverage\_report.html

````

4. **Copy the report back to the host**
- Find the container ID:
  ```bash
docker ps
  ```
- Copy the report:
  ```bash
docker cp <container_id>:/usr/local/src/opencv/build/coverage_report.html .
  ```
- Open the report:
  - macOS: `open ./coverage_report.html`
  - Windows: `start .\coverage_report.html`
  - Linux: `xdg-open ./coverage_report.html`

---

## Generate Coverage (gcovr)

1. **Enter the OpenCV build directory (inside the container)**

   ```bash
   ```

cd /usr/local/src/opencv/build/

````

2. **Install gcovr**
```bash
pip install gcovr
````

3. **Generate an HTML coverage report**

   ```bash
   ```

gcovr -r /usr/local/src/opencv --html --html-details -o coverage\_report.html

````

4. **Copy the report back to the host**
- Find the container ID:
  ```bash
docker ps
  ```
- Copy the report:
  ```bash
docker cp <container_id>:/usr/local/src/opencv/build/coverage_report.html .
  ```
- Open the report:
  - macOS: `open ./coverage_report.html`
  - Windows: `start .\coverage_report.html`
  - Linux: `xdg-open ./coverage_report.html`

---

## Standardized API
VistaFuzz consumes standardized API metadata to guide input generation:
````

OpenCV-Testing/API\_info.py

````
This file describes each API's **name, parameters, types/constraints**, etc., which the fuzzer uses to synthesize valid and diverse inputs.

---

## Run without Docker
> **Not recommended** (you may encounter compiler/coverage toolchain differences). If necessary, ensure your local environment matches the container and consider the following:

- Python ≥ 3.8; gcc/clang supports coverage flags (`--coverage` or `-fprofile-arcs -ftest-coverage`).
- Install `gcovr` (or `lcov`/`genhtml`).
- Build OpenCV from source with coverage enabled, ensuring `.gcno/.gcda` files are produced under `build`.
- Pin key package versions (example):
  ```bash
  pip install "numpy==1.26.*" "gcovr==7.*"
````

* Minor differences in BLAS/hardware may lead to small coverage or numerical deviations.

---

## Expected Outputs

* Running `python3 main.py` outputs which APIs/test cases were executed and runtime logs.
* Running `gcovr` produces an HTML coverage report at:

  ```
  /usr/local/src/opencv/build/coverage_report.html
  ```

  Copy it to your host and open it in a browser. The report includes line-by-line coverage with source highlighting.

> Note: Coverage values depend on runtime and the number of generated test cases; the primary verification goal is **successful report generation** and **navigable source coverage**.

---

## Cite This Artifact

* **DOI:** [10.5281/zenodo.16891964](https://doi.org/10.5281/zenodo.16891964)

<details>
<summary><b>BibTeX</b> (click to expand)</summary>

```bibtex
@software{vistafuzz_zenodo_16891964,
  title   = {VistaFuzz: Documentation-guided fuzzing for OpenCV},
  author  = {Duan, Bin and Dong, Ruican and Kim, Dan Dongseong and Yang, Guowei},
  year    = {2025},
  doi     = {10.5281/zenodo.16891964},
  url     = {https://doi.org/10.5281/zenodo.16891964}
}
```

</details>

---

## License

This project is released under the [MIT License](LICENSE).
