# Mat-mul

## Build
`make`

## Usage

### OpenMP

To run the OpenMP code, you'll need to run the follow command to increase the stack size

```
ulimit -s unlimited
export OMP_STACKSIZE=500m

./mat_mut_openmp2 1024
```

Perf Cache misses for multithreads
```
sudo su
sysctl -w kernel.perf_event_paranoid=-1
sysctl kernel.nmi_watchdog=0

perf stat -B -e cache-references,cache-misses,cycles,instructions,branches,faults,migrations sleep 5

perf record -e page-faults -a -g -- sleep 60
perf script --header > out.stack

./mat_mul & echo $! 
perf record -F 99 -p 181 -g -- sleep 60
perf script > out.perf

./mat_mul_pt3_stride 4098 & perf record -F 99 -p $! -g -- sleep 60 

```