template<class T>
T add(const T a, const T b) {
    return a + b;
}

__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {

    // Get the index of the current element to be processed
    int i = get_global_id(0);

    // Do the operation
    C[i] = add(A[i], B[i]);
}