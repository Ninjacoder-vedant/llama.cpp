// New backend for GGML that uses XLNS16 ops
// Copyright (c) 2026 Vedant Acharya

// Description:
//   A virtual hardware backend for ggml that implements the Logarithmic Number 
//   System (LNS). This backend intercepts standard floating-point operations 
//   (specifically GGML_OP_MUL_MAT) and processes them using 16-bit 
//   LNS arithmetic via the xlnscpp library. 
// 
//   This serves as a software Proof of Concept (PoC) for LNS-based Large 
//   Language Model inference.

#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-xlns.h"

#define xlns16_alt
#define xlns16_table
#include "xlns16.cpp"

#include <string>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

static const char * ggml_backend_xlns_get_name(ggml_backend_t backend) {
    return "xlns";

    GGML_UNUSED(backend);
}

static void ggml_backend_xlns_free(ggml_backend_t backend) {
    delete backend; // There's no context so only backend should be freed
} 

void matmul_xlns16_float(const float *A, const float *B, float *C, int64_t M, int64_t N, int64_t K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            xlns16_float sum = float2xlns16_(0.0f);

            for (int k = 0; k < N; k++)
            {
                sum += float2xlns16_(A[i * N + k]) * float2xlns16_(B[j * N + k]);
            }
            C[j * M + i] = xlns16_2float(sum);
        }
    }
}

static void ggml_backend_xlns_mul_mat(struct ggml_tensor * dst) {
    static bool has_printed = false;
    if (!has_printed) {
        printf("\n");
        printf("\033[1;36m============================================================\033[0m\n");
        printf("\033[1;32m      🚀 XLNS-16 VIRTUAL HARDWARE ACCELERATOR ONLINE 🚀     \033[0m\n");
        printf("\033[1;36m============================================================\033[0m\n");
        printf("\033[1;33m  >>> Intercepting GGML_OP_MUL_MAT with Logarithmic Math <<<\033[0m\n\n");
        has_printed = true; // Never print this again for the rest of the run
    }
    
    const struct ggml_tensor * src0 = dst->src[0]; 
    const struct ggml_tensor * src1 = dst->src[1]; 

    // Extract ggml dimensions
    const int64_t ggml_K = src0->ne[0]; // Inner dimension
    const int64_t ggml_M = src0->ne[1]; // Rows in A
    const int64_t ggml_N = src1->ne[1]; // Columns in B (Batch size)

    // Extract raw memory pointers
    const float * A = (const float *) src0->data;
    const float * B = (const float *) src1->data;
    float       * C = (float *)       dst->data;

    // M = rows (ggml_M), N = inner dim (ggml_K), K = batch (ggml_N)
    matmul_xlns16_float(A, B, C, ggml_M, ggml_K, ggml_N);
}

static enum ggml_status ggml_backend_xlns_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    
    // Iterate through every operation in the computation graph linearly
    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        struct ggml_tensor * node = cgraph->nodes[node_idx];

        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                ggml_backend_xlns_mul_mat(node);
                break;
            }

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            // case GGML_OP_ADD: {
            //     // You would put a similarly simple 1D array loop here for addition
            //     break;
            // }

            default: {
                // If ggml somehow hands us an operation we didn't authorize in 
                // supports_op, fail gracefully.
                printf("[XLNS FATAL] Unhandled operation routed to backend: %s\n", ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
            }
        }
    }

    // We successfully completed the entire graph!
    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

static const struct ggml_backend_i ggml_backend_xlns_i = {
    /* .get_name                = */ ggml_backend_xlns_get_name,
    /* .free                    = */ ggml_backend_xlns_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_xlns_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static const char * ggml_backend_xlns_device_get_name(ggml_backend_dev_t dev) {
    return "xlns";

    GGML_UNUSED(dev);
}

// struct ggml_backend_cpu_device_context {
//     std::string description = "xlns16 Virtual Machine";
// };

static const char * ggml_backend_xlns_device_get_description(ggml_backend_dev_t dev) {
    return "xlns16 Virtual Machine";

    GGML_UNUSED(dev);
}

static void ggml_backend_xlns_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free  = status.ullAvailPhys;
#else
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total         = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *free = *total;
#endif  // _WIN32

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xlns_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static ggml_guid_t ggml_backend_xlns_guid(void) {
    // A completely unique, randomly generated 16-byte ID for the LNS backend.
    static ggml_guid guid = { 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x9a,
                              0xbc, 0xde, 0xf1, 0x23, 0x45, 0x67, 0x89, 0xab };
    return &guid;
}

static ggml_backend_t ggml_backend_xlns_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    // can init xlns backend now to avoid slowing the first graph computation
    // xlns_init()

    ggml_backend_t xlns_backend = new ggml_backend{
        /* .guid    = */ ggml_backend_xlns_guid(),
        /* .iface   = */ ggml_backend_xlns_i,
        /* .device  = */ dev,
        /* .context = */ NULL,  // No need for context state for now
    };
    return xlns_backend;

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static bool ggml_backend_xlns_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {

    // Only accept the math operations that are actually coded in LNS compute graph
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;        

        default:
            // For EVERYTHING ELSE (Softmax, RMSNorm, Log, SiLU, etc.) return FALSE!
            // This safely forces ggml to fall back to the CPU for these operations.
            return false;
    }
    GGML_UNUSED(dev);
}

static bool ggml_backend_xlns_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_xlns_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static void ggml_backend_xlns_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xlns_device_get_name(dev);
    props->description = ggml_backend_xlns_device_get_description(dev);
    props->type        = ggml_backend_xlns_device_get_type(dev);
    ggml_backend_xlns_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static const struct ggml_backend_device_i ggml_backend_xlns_device_i = {
    /* .get_name             = */ ggml_backend_xlns_device_get_name,
    /* .get_description      = */ ggml_backend_xlns_device_get_description,
    /* .get_memory           = */ ggml_backend_xlns_device_get_memory,
    /* .get_type             = */ ggml_backend_xlns_device_get_type,
    /* .get_props            = */ ggml_backend_xlns_device_get_props,
    /* .init_backend         = */ ggml_backend_xlns_device_init_backend,
    /* .get_buffer_type      = */ [](ggml_backend_dev_t) { return ggml_backend_cpu_buffer_type(); },
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_xlns_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xlns_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xlns_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static const char * ggml_backend_xlns_reg_get_name(ggml_backend_reg_t reg) {
    return "xlns";

    // included in ggml.h, because reg is not used, compiler may give warning so this is to suppress the warning
    GGML_UNUSED(reg);
}

static size_t ggml_backend_xlns_reg_get_device_count(ggml_backend_reg_t reg) {
    // We only have 1 virtual LNS device running on the CPU
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_xlns_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    // We use 'static' so the device persists in memory for the life of the program
    static struct ggml_backend_device ggml_backend_xlns_device = {
        /* .iface   = */ ggml_backend_xlns_device_i,  // The device contract we made earlier
        /* .reg     = */ reg,
        /* .context = */ NULL,                        // You can link your buffer type here if needed (Just a cpu model)
    };

    return &ggml_backend_xlns_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static const struct ggml_backend_reg_i ggml_backend_xlns_reg_i = {
    /* .get_name         = */ ggml_backend_xlns_reg_get_name,
    /* .get_device_count = */ ggml_backend_xlns_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xlns_reg_get_device,
    /* .get_proc_address = */ NULL,  // Not needed for our LNS PoC
};

// The Main Initialization Function
ggml_backend_reg_t ggml_backend_xlns_reg(void) {
    // If xlnscpp requires any one-time setup (like generating lookup tables),
    // we can call that init function right here.
    // xlns_init();

    static struct ggml_backend_reg ggml_backend_xlns_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xlns_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_xlns_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xlns_reg) // For dynamic linking