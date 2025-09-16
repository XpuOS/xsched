__kernel void vector_add(__global const float *src_a,
                         __global const float *src_b,
                         __global float *res,
                         int num) {
  // Get_global_id (0) returns the ID of the executing thread.
  // Many threads will start executing the same kernel at the same time,
  // Each thread will receive a different ID, so it will inevitably perform a
  // different computation.
  int idx = get_global_id(0);

  // Each work item checks if its ID is within the range of the vector array.
  // If so, the work item will perform the corresponding calculations.
  if (idx < num)
    res[idx] = src_a[idx] + src_b[idx];
}