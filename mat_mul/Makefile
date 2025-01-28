# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_89

# Directories
INCLUDE_DIR = include 
KERNELS_DIR = kernels
BUILD_DIR = build

# Automatically detect all .cu files
KERNEL_SOURCES = $(wildcard $(KERNELS_DIR)/*.cu)
ROOT_SOURCES = src.cu
CUDA_SOURCES = $(KERNEL_SOURCES) $(ROOT_SOURCES) 

# Generate object file names
CUDA_OBJECTS = $(patsubst $(KERNELS_DIR)/%.cu,$(BUILD_DIR)/%.obj,$(KERNEL_SOURCES)) $(BUILD_DIR)/src.obj

# Output executable
OUTPUT = matmul_benchmark

# Rules
all: $(OUTPUT)

$(OUTPUT): $(CUDA_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Rules for compiling kernels directory .cu files
$(BUILD_DIR)/%.obj:$(KERNELS_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Rules for compiling root directory .cu files
$(BUILD_DIR)/src.obj: src.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Create the build directory
$(BUILD_DIR):
	mkdir $(BUILD_DIR)

# Clean Rule
clean:
	rm $(BUILD_DIR)/*.obj $(OUTPUT) 

.PHONY: all clean