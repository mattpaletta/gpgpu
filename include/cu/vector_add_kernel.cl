__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
    /* These comments must be closed because they will get appended to one line */

    /* Get the index of the current element to be processed*/
    int i = get_global_id(0);

    /* Do the operation */
    C[i] = A[i] + B[i];
}