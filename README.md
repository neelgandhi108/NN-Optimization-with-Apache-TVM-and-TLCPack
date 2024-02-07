
# NN Optimization with Apache-TVM
![NN Optimization]("Assets/LP_Pipeline.jpg")

## Setup

For a straightforward installation of TVM, leveraging the [TLCPack](https://tlcpack.ai/) is recommended. Note, however, that installations via TLCPack do not support CUDA experiments. Additionally, extremely recent versions of Python might be incompatible with TLCPack, though it is confirmed to work with Python 3.7.

For manual installation, refer to the [source installation guide](https://tvm.apache.org/docs/install/from_source.html#install-from-source). Crucially, you must adjust the _config.cmake_ file to tailor the installation to your needs. This file is located within the _tvm/cmake_ directory. Follow the provided instructions to accurately edit it.

## TVM Summary

![TVM Architecture]("Assets\TVMStack.jpg")

Each stage in a TVM workflow corresponds to a specific class.

**Load**: At this stage, a model from a compatible framework is imported and transformed into a relay format. The outcome is a TVMCModel, encapsulating the pre-compiled graph and model parameters.

**Tune**: The tuning phase involves computing expressions, which outline the operations and output computations, and scheduling, which suggests modifications to these expressions for optimization. TVM optimizes by dissecting the model into workloads, optimizing each, then choosing the best schedule per workload, resulting in a tuning report where lower latency and higher GFLOPS indicate better performance.

**Compile**: This phase converts a TVMCModel into a TVMCPackage, containing all necessary components to execute the model on the designated hardware. More details on targets can be found [here](https://tvm.apache.org/docs/reference/api/python/target.html).

**Run**: Executing a TVMCPackage yields a TVMCResult, detailing model outputs and execution time. Input data can be specified via the _inputs_ parameter, with options for auto-generating inputs if none are provided.

## Project Layout

-   _main.py_ houses the experiment execution code.
-   _utils/argparser.py_ contains a command-line argument parser. Use `python3 main.py --help` for options.
-   Download the onnx pre-trained resnet50 model using:
    
    consoleCopy code
    
    `wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx` 
    
    Place it in the _models_ directory for use.

Results from tuning are stored in the _logs_ folder. The script _LP_autoTVM.py_ facilitates model optimization and runtime comparison on TVM. For TensorFlow examples and performance comparisons, refer to _tf_example.py_ and _tf_performances_ respectively.

## Performance Benchmarks

Benchmark results for various models, both pre- and post-optimization, are tabulated below. Notably, the keras models initially run slower on TVM but achieve superior optimization results, whereas the .onnx models consistently exhibit lower variance in execution times, both pre- and post-optimization. Further experimentation with different models might elucidate these trends

![Results1]("Assets\TVM1.jpg")
![Results2]("Assets\TVM2.jpg")
![Results3]("Assets\TVM3.jpg")
