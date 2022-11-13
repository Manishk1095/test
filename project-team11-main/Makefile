build: build_matmul build_block build_transpose build_unroll build_pt build_rdpmc build_openmp

build_matmul:
	gcc -o mat_mul mat_mul.c

build_block:
	gcc -o mat_mul_block mat_mul_block.c

build_transpose:
	gcc -o mat_mut_transposed mat_mut_transposed.c

build_unroll:
	gcc -o mat_mul_unroll mat_mul_unroll.c

build_pt:
	gcc -o mat_mul_pt_naive mat_mul_pt_naive.c
	gcc -fopenmp -o mat_mul_pt mat_mul_pt.c
	gcc -fopenmp -o mat_mul_pt2_precopy mat_mul_pt2_precopy.c
	gcc -fopenmp -o mat_mul_pt3_stride mat_mul_pt3_stride.c

build_rdpmc:
	gcc -o mat_mul_rdpmc mat_mul_rdpmc.c

build_openmp:
	gcc -fopenmp -o mat_mut_openmp1 mat_mut_openmp1.c
	gcc -fopenmp -o mat_mut_openmp2 mat_mut_openmp2.c

clean:
	rm -f *.o*
	rm -f mat_mul
	rm -f mat_mul_block
	rm -f mat_mut_transposed
	rm -f mat_mul_unroll
	rm -f mat_mul_pt_naive mat_mul_pt mat_mul_pt2_precopy mat_mul_pt3_stride
	rm -f mat_mul_rdpmc
	rm -f mat_mut_openmp1 mat_mut_openmp2
