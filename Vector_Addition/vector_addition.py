# basics
# test
# benchmark

# importing libraries

import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda : {torch.cude.current_device()}')

# We will be mostly using the triton.language library to perform all the triton operations.

# The @triton.jit decorator is used to mark a function as a kernel that can be executed on the GPU.
@triton.jit
def add_kernel(
    x_ptr , y_ptr , output_ptr , num_elements , BLOCK_SIZE : tl.constexpr
):
  
  # tl.program_id gives each kernel a unique id. 'Grid' is the number of pids generated. 
  pid = tl.program_id(axis = 0)
  block_start = pid*BLOCK_SIZE
  # block_start will be --> 0*1024 , 1*1024 , 2*1024 ... n*1024
  offsets = block_start + tl.arange(0 , BLOCK_SIZE)
  # offets ensure traversal of all the elements that a worker has to handle for each pid. 
  mask = offsets<num_elements

  # mask looks something like this :- [True , True , False , False]
  # The true values are the one's where the operaion will be performed and the false values will be ignored. 
  # This helps in saving GPU resources and time as they are not needed. 

  x = tl.load(x_ptr+offsets , mask = mask , other = None)
  y = tl.load(y_ptr+offsets , mask = mask , other = None)

  output = x+y
  tl.store(output_ptr+offsets , output , mask = mask)

def add(x , y):
  # predefining output
  output = torch.empty_like(x)
  assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

  num_elements = output.numel()
  grid = lambda meta : (triton.cdiv(num_elements , ['BLOCK_SIZE']) , )

  # BLOCK_SIZE --> How many elements a worker has to handle
  
  add_kernel[grid](
      x , y , output , num_elements , BLOCK_SIZE = 1024
  )

  # When we pass 'grid' in the "[]" , we are not just passing a random variable to the function but are configuring the GPU Scheduler.
  # The GPU driver launches 'add_kernel' 'grid' number of times.
  # The Logic Flow:
  # | Program Instance | pid | block_start (pid * 1024) | offsets (indices) |
  # | :--- | :--- | :--- | :--- |
  # | Worker 0 | 0 | 0 | 0, 1, 2, ... 1023 |
  # | Worker 1 | 1 | 1024 | 1024, 1025, ... 2047 |
  # | Worker 2 | 2 | 2048 | 2048, 2049, ... 3071 |
  # | Worker 3 | 3 | 3072 | 3072, 3073, ... 4095 |


  return output


def test_add_kernel(size , atol = 1e-3 , rtol = 1e-3):
  # for testing the wrapper function and our vector addition kernel 

  # input data
  x = torch.randn(size , device = DEVICE)
  y = torch.randn(size , device = DEVICE)
  
  # pytorch reference and kernel function
  z_torch = x+y
  z_triton = add(x,y)

  # comparison
  torch.testing.assert_close(z_triton , z_torch , atol = atol , rtol = rtol)
  print("PASSED")



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(12, 28, 1)], # different values of x_names to benchmark
        x_log = True, # logarithmic scale for x-axis
        line_arg='provider', # title of the legend 
        line_vals=['triton', 'torch'], # designators of the different entries in the legend
        line_names=['Triton', 'Torch'], # names to visibly go in the legend
        styles=[('blue', '-'), ('green', '-')], # triton will be blue; pytorch will be green
        ylabel='GB/s', # label name for y-axis
        plot_name='vector-add-performance', # also used as file name for saving plot
        args={}, # any extra arguments to pass to the benchmark function
    )
)
def benchmark(size, provider):
    # creating our input data
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    # each benchmark runs multiple times and quantiles 
    quantiles = [0.5, 0.05, 0.95]
    # defining which function this benchmark instance runs
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    # turning the raw millisecond measurement into meaninful units
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":

    # Unit Tests
    test_add_kernel(size=1024)
    test_add_kernel(size=1025)

    # Primary Use Case
    test_add_kernel(size=98432)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)