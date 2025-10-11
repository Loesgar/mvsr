# Fast Min-ε Segmented Regression using Constant-Time Segment Merging

This branch contains the original evaluation code for the ICML paper.
**If you want to use our algorithm yourself, consider using our library, check out the [main project site](https://github.com/Loesgar/mvsr).**

The extended evaluation regarding parameter $d$ ~~can be found [in the evald folder](evald/evaluation.pdf)~~ is now part of the appendix of the [final camera-ready version of the paper](https://proceedings.mlr.press/v267/losser25a.html).

---------------------------------------------------------------------

This repository enables the execution of the evaluation. To achieve this, there are two separate steps involved:

1. Measurement (executing the regression algorithms)
2. Analysis (using the generated CSV files to generate plots)

This folder contains our source files in the root folder. Additionally, the folders contain the following data:

| Folder        | Explanation                                                                                                                                                                                                                                               |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| build/        | The build folder contains the algorithm-binary we used. This might not work on your system (e.g., because of the clib version). If it does not work you can build it yourself (see below). **Delete this folder first** if you want to build it yourself. |
| data/         | This folder contains the raw time series data we used for own evaluation. It also includes a script to convert our CSV to a file that can be used as input in our ```benchmark.py``` script.                                                              |
| eigen/        | A matrix-/linalg-library we used when implementing our approach (only exists after cloning eigen, see *Download Dependencies*).                                                                                                                           |
| eval/         | Our evaluation results (dataset). We also included our generated graphs here.                                                                                                                                                                             |
| evald/        | Evaluation results (dataset) of the supplemental material. We also included our generated graphs here.                                                                                                                                                    |
| related_work/ | The source code of the algorithms we compared our approach to.                                                                                                                                                                                            |

## Measurement

These steps will prepare and execute the measurement.

### System Setup

We did our measurements on an x86_64 CPU. On Ubuntu 24.04 LTS, the necessary packages can be installed by executing the following shell commands. On a container running Ubuntu 24.04 this should also work.

```sh-session
sudo apt install cmake g++ gcc git make python3 python3-pip python3-venv
```

### Download Dependencies

This downloads the *eigen* library we used for our implementation, as well as the source code of the other algorithms in the related_work folder. This is an exact mirror of the open github repositories. We used git to download the code for the other algorithms.

```sh-session
cd <segmented-regression>    # change working dir to the folder containing this README.md
./fetch-dependencies.sh
```

### Building the new Heuristic

These steps are needed to build our algorithm. In our case the used compiler was gcc *13.2.0*.

```sh-session
rm -rf build
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
cd ..
```

### Evaluation Setup

We need to setup a python virtual environment and install the needed packages.

```sh-session
python3 -m venv .venv
source ./.venv/bin/activate
python3 -m pip install -r requirements.txt
```

To enable the virtual environment again after installation, execute the following code. This contains also all packages needed for the plot generation.

```sh-session
source ./.venv/bin/activate
```

### Evaluation Execution

The following commands execute the measurement. **Attention:** this can take quite a while. On our machine the measurement took roughly 24h. You can reduce the number of repetitions or the number of data points to reduce the execution time. The ``python3 benchmark.py --help`` command prints the usage documentation.

If you are planning to do this over an ssh connection you should consider using ``screen`` or ``tmux`` in order to be able to execute in the background, even if the connection drops.

#### Synthetic Data

```sh-session
mkdir eval
cd eval
/usr/bin/time python3 ../benchmark.py --sigmas 0.01 0.1 --repeats 100 --seed 1 --range 6 24 --algorithms PREG 2>&1 | tee out_preg.txt
/usr/bin/time python3 ../benchmark.py --sigmas 0.01 0.1 --repeats 100 --seed 1 --range 6 18 --algorithms M1K M2K M4K 2>&1 | tee out_fsm.txt
/usr/bin/time python3 ../benchmark.py --sigmas 0.01 0.1 --repeats 100 --seed 1 --range 6 14 --algorithms DP 2>&1 | tee out_dp.txt

cat /proc/cpuinfo > cpu.txt
uname -srvmpio > os.txt
cat /etc/lsb-release >> os.txt
cd ..
```

The resulting files are saved inside the working directory and can be used in the next steps. Our own measurements are located in the folder ``eval``.

#### Real Data

Our time series measurements of the execution of ```bwa``` are saved in the file ```data/task_948_bwa__DA45_.csv```.
We used the script ```data/convert_data.py``` to extract the CPU and memory data and save it as separate files (also in the ```data``` folder).
These files are compatible to be used as input for ```benchmark.py```.

```sh-session
cd eval
python3 ../benchmark.py --txt ../data/task_948_bwa__DA45_CPU.txt --target-segments 4 --algorithms DP M1K M2K M4K PREG
cd ..
```

These will again generate CSV files with the regression results for the CPU curve.
Multiple things should be noted:

1. All segment counts from 2 to 4 are evaluated. We ignore all versions that are smaller than 4.
2. Every execution is done 8 times.
3. We do not know the ground truth of the functions. Therefore, the error relative to the ground truth is *NAN*.

#### Evaluating the Number of Dimensions

The following is needed for the evaluation in the supplementary material. We included two patchsets for Acharya et al. to either (a) fix the rank-1 update of the DP to be $d^2$ or (b) always directly solving the linear system ($d^3$ but more stable).

```sh-session
mkdir evald
cd evald
mkdir n4096 n8192 n64d
cd ..

cd related_work/fast_segmented_regression/
git apply ../acharya_sm_repair.patch
cd ../..

cd evald/n4096
python ../../benchmark.py --algorithms DP --dimensions 2 4 8 16 32 64 --ns 4096 --no-nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms DP --dimensions 2 4 8 16 32 64 --ns 4096 --no-nd --sigmas 0.1 --repeats 100 --target-segments 4
for i in DP_CONT_*.csv; do mv $i DPSM${i:2}; done

cd ../n8192
python ../../benchmark.py --algorithms DP --dimensions 2 4 8 16 32 64 --ns 8192 --no-nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms DP --dimensions 2 4 8 16 32 64 --ns 8192 --no-nd --sigmas 0.1 --repeats 100 --target-segments 4
for i in DP_CONT_*.csv; do mv $i DPSM${i:2}; done

cd ../n64d
python ../../benchmark.py --algorithms DP --dimensions 2 4 8 16 32 64 --ns 64 --nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms DP --dimensions 2 4 8 16 32 64 --ns 64 --nd --sigmas 0.1 --repeats 100 --target-segments 4
for i in DP_CONT_*.csv; do mv $i DPSM${i:2}; done
cd ../..

cd related_work/fast_segmented_regression/
git reset --hard
git apply ../acharya_sm_disable.patch
cd ../..

cd evald/n4096
python ../../benchmark.py --algorithms DP --dimensions 2 4 8 16 32 64 --ns 4096 --no-nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms DP --dimensions 2 4 8 16 32 64 --ns 4096 --no-nd --sigmas 0.1 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms M1K M2K M4K --dimensions 2 4 8 16 32 64 128 --ns 4096 --no-nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms M1K M2K M4K --dimensions 2 4 8 16 32 64 128 --ns 4096 --no-nd --sigmas 0.1 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms PREG --dimensions 2 4 8 16 32 64 128 256 --ns 4096 --no-nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms PREG --dimensions 2 4 8 16 32 64 128 256 --ns 4096 --no-nd --sigmas 0.1 --repeats 100 --target-segments 4

cd ../n8192
python ../../benchmark.py --algorithms DP --dimensions 2 4 8 16 32 64 --ns 8192 --no-nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms DP --dimensions 2 4 8 16 32 64 --ns 8192 --no-nd --sigmas 0.1 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms M1K M2K M4K --dimensions 2 4 8 16 32 64 128 --ns 8192 --no-nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms M1K M2K M4K --dimensions 2 4 8 16 32 64 128 --ns 8192 --no-nd --sigmas 0.1 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms PREG --dimensions 2 4 8 16 32 64 128 256 --ns 8192 --no-nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms PREG --dimensions 2 4 8 16 32 64 128 256 --ns 8192 --no-nd --sigmas 0.1 --repeats 100 --target-segments 4

cd ../n64d
python ../../benchmark.py --algorithms DP --dimensions 2 4 8 16 32 64 --ns 64 --nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms DP --dimensions 2 4 8 16 32 64 --ns 64 --nd --sigmas 0.1 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms M1K --dimensions 2 4 8 16 32 64 128 --ns 64 --nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms M1K --dimensions 2 4 8 16 32 64 128 --ns 64 --nd --sigmas 0.1 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms M2K --dimensions 2 4 8 16 32 64 128 --ns 64 --nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms M2K --dimensions 2 4 8 16 32 64 128 --ns 64 --nd --sigmas 0.1 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms M4K --dimensions 2 4 8 16 32 64 128 --ns 64 --nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms M4K --dimensions 2 4 8 16 32 64 128 --ns 64 --nd --sigmas 0.1 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms PREG --dimensions 2 4 8 16 32 64 128 256 --ns 64 --nd --sigmas 0.01 --repeats 100 --target-segments 4
python ../../benchmark.py --algorithms PREG --dimensions 2 4 8 16 32 64 128 256 --ns 64 --nd --sigmas 0.1 --repeats 100 --target-segments 4

cd ../..
```

### Notes

After the first execution the Julia *StatsBase* package is installed. The following lines are already inside the ``benchmark.py`` file. You can comment them out and execute the script again. This is faster for small future execution runs.

```python
import juliapkg
juliapkg.add("StatsBase", "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91")
juliapkg.resolve()
```

For real data that is analysed with the DP algorithm it might make sense to reduce the repetitions or adapt the script to not consider the smaller target_segments values.
In our setup, every execution needed over 40 minutes.

## Analysis

We describe how to generate the plots and numbers from our paper based on our measurements from the eval folder.
The easiest method to do the same evaluation with new measurements would be to **delete our data from the eval folder** and copy the CSV files from the own measurements to the eval folder.
After the above steps, this data is inside the main folder, which also contains this README file.

If using the virtual python environment from the above steps, all libraries should be installed correctly to execute the plotting scripts.

### Synthetic Data

Execute the ``plot.py`` script and provide the path to the *evaln* folder.

```sh-session
cd eval
python3 ../plot.py .
```

The plots will be shown and also saved as separate PDF files in the current working directory.
It will also print the average relative error of all algorithms relative to the DP-algorithm for $\sigma = 0.1$.

Similarly the plot to evaluate parameter $d$ can be generated.

```sh-session
cd evald
python3 ../plotd.py n4096 n8192 n64d
```

### Real Data

Plotting the real data analysis needs (a) the original data points and (b) regression results.
Our ``plot_real.py`` script needs the original data used for ``benchmark.py`` as an input and searches the working directory for the corresponding CSV files (if no second parameter is given).
It will plot and save the figure.

```sh-session
cd eval
python3 ../plot_real.py ../data/task_948_bwa__DA45_CPU.txt
cd ..
```

The numbers from our table can be read directly from the CSV files.
Since the algorithms are deterministic, the resulting regressions were always the same and can be read (together with the errors) from the CSV directly.
The execution times should in theory be exactly equal as well, since the same data was analysed.
In our case, there were no meaningful variations when analyzing the same data.

A common practice in this specific scenario is to take the smallest time measure, since all outliers like scheduler interruptions should only increase the time and the smallest value would therefore be closest to an uninterrupted execution of the algorithm.
**This should only be done if there are no big outliers with small execution times.**
We checked our data regarding this.
As expected, all time values are very close, despite occasional outliers with high execution times.
In such scenario, the smallest execution times (with $k=4$) for the algorithms can be listed with the following command.

```sh-session
cd eval
for alg in DP M1K M2K M4K PREG; do echo -n "$alg: "; grep ',4,' ${alg}_CSV_* | awk -F "," '{print $8}' | sort | head -n1; done
cd ..
```

Similarly we can output the errors, dividing them with the DP yields the relative error increase.

```sh-session
cd eval
for alg in DP M1K M2K M4K PREG; do echo -n "$alg: "; grep ',4,' ${alg}_CSV_* | tail -n1 | awk -F "," '{print $10}'; done
cd ..
```
