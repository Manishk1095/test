#!/bin/bash
export KMP_STACKSIZE=1g
export OMP_STACKSIZE=1g
ulimit -s unlimited 
./mat_mul_pt 256 0 >> mat_mul_pt_taff.log
./mat_mul_pt 256 0 >> mat_mul_pt_taff.log
./mat_mul_pt 256 0 >> mat_mul_pt_taff.log
./mat_mul_pt 512 0 >> mat_mul_pt_taff.log
./mat_mul_pt 512 0 >> mat_mul_pt_taff.log
./mat_mul_pt 512 0 >> mat_mul_pt_taff.log
./mat_mul_pt 1024 0 >> mat_mul_pt_taff.log
./mat_mul_pt 1024 0 >> mat_mul_pt_taff.log
./mat_mul_pt 1024 0 >> mat_mul_pt_taff.log
./mat_mul_pt 2048 0 >> mat_mul_pt_taff.log
./mat_mul_pt 2048 0 >> mat_mul_pt_taff.log
./mat_mul_pt 2048 0 >> mat_mul_pt_taff.log
./mat_mul_pt 4096 0 >> mat_mul_pt_taff.log
./mat_mul_pt 4096 0 >> mat_mul_pt_taff.log
./mat_mul_pt 4096 0 >> mat_mul_pt_taff.log

./mat_mul_pt2_precopy 256 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 256 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 256 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 512 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 512 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 512 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 1024 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 1024 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 1024 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 2048 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 2048 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 2048 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 4096 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 4096 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 4096 0 >> mat_mul_pt2_precopy_taff.log

./mat_mul_pt3_stride 256 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 256 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 256 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 512 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 512 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 512 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 1024 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 1024 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 1024 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 2048 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 2048 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 2048 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 4096 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 4096 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 4096 0 >> mat_mul_pt3_stride_taff.log

./mat_mul_pt 8192 0 >> mat_mul_pt_taff.log 
./mat_mul_pt 8192 0 >> mat_mul_pt_taff.log
./mat_mul_pt 8192 0 >> mat_mul_pt_taff.log

./mat_mul_pt2_precopy 8192 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 8192 0 >> mat_mul_pt2_precopy_taff.log
./mat_mul_pt2_precopy 8192 0 >> mat_mul_pt2_precopy_taff.log

./mat_mul_pt3_stride 8192 0 >> mat_mul_pt3_stride_taff.log 
./mat_mul_pt3_stride 8192 0 >> mat_mul_pt3_stride_taff.log
./mat_mul_pt3_stride 8192 0 >> mat_mul_pt3_stride_taff.log
