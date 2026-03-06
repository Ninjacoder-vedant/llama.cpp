### XLNS: 16-bit Logarithmic Number System (LNS) Virtual Backend

#### Overview
The goal of this xlns backend PoC is to demonstrate the viability of LNS-based LLM inference. By intercepting standard floating-point operations and processing them via LNS, this backend serves as a software sandbox for researching LNS hardware acceleration.

#### File structure

```bash
ggml/
├── include/
│   └── ggml-xlns.h          # header file
└── src/
    └── ggml-xlns/
        ├── CMakeLists.txt
        ├── ggml-xlns.cpp    # main file
        ├── xlns16.cpp
        ├── xlns16cvtbl.h
        ├── xlns16exptbl.h
        ├── xlns16logtbl.h
        ├── xlns16revcvtbl.h
        └── xlns16sbdbtbl.h`
```

#### Architecture
This initial implementation acts as an `ACCEL` device working alongside the CPU:

* **Zero-Copy Memory:** It uses `ggml_backend_cpu_buffer_type()` to share host RAM directly with the CPU backend, avoiding expensive VRAM transfers (similar to BLAS).
* **Targeted Interception:** The backend explicitly intercepts `GGML_OP_MUL_MAT` where all input tensors are `GGML_TYPE_F32`.
* **Safe Fallback:** All other operations (Softmax, Add, Norm) and View operations are gracefully ignored or passed back to the standard CPU backend.
* **Math Loop:** Inside the compute graph, standard `FP32` data is cast to `xlns16_float`, multiplied and accumulated using LNS arithmetic, and converted back to `FP32`.

#### How to Build

A new CMake option has been added. You can build the backend from root of the repo by enabling the `GGML_XLNS` flag:

```bash
cmake -B build -DGGML_XLNS=ON
cmake --build build --config Release -j 8
```

#### How to Test
Because this backend specifically targets `F32` matrix multiplications, it requires an unquantized `F32` model. I recommend using a tiny model like Karpathy's 15M Stories model to see the math loop in action quickly:

```bash
./build/bin/llama-cli \
  --hf-repo ggml-org/models \
  --hf-file tinyllamas/stories15M-f32.gguf \
  -p "Once upon a time, in a small village, there lived a little girl named" \
  -n 128 \
  --temp 0
```