
#pragma once
#include "simulator.hpp"
#include <vector>
#include <cassert>
#include <string>

namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  Matrix *acc_K = nullptr;
  Matrix *acc_V = nullptr;
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(values[i]);

    if (i == 0) {
      acc_K = keys[i];
      acc_V = values[i];
    } else {
      Matrix *new_K = matrix_memory_allocator.Allocate("acc_K_" + std::to_string(i));
      gpu_sim.Concat(acc_K, keys[i], new_K, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(acc_K);
      gpu_sim.ReleaseMatrix(keys[i]);
      acc_K = new_K;

      Matrix *new_V = matrix_memory_allocator.Allocate("acc_V_" + std::to_string(i));
      gpu_sim.Concat(acc_V, values[i], new_V, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(acc_V);
      gpu_sim.ReleaseMatrix(values[i]);
      acc_V = new_V;
    }

    // S = Q * K^T
    Matrix *S = nullptr;
    for (size_t j = 0; j < 512; ++j) {
      Matrix *q_col = matrix_memory_allocator.Allocate("q_col");
      gpu_sim.GetColumn(current_query, j, q_col, kInSharedMemory);
      Matrix *k_col = matrix_memory_allocator.Allocate("k_col");
      gpu_sim.GetColumn(acc_K, j, k_col, kInSharedMemory);
      gpu_sim.Transpose(k_col, kInSharedMemory);
      Matrix *prod = matrix_memory_allocator.Allocate("prod");
      gpu_sim.MatMul(q_col, k_col, prod);
      if (S == nullptr) {
        S = prod;
      } else {
        Matrix *new_S = matrix_memory_allocator.Allocate("S_acc");
        gpu_sim.MatAdd(S, prod, new_S);
        gpu_sim.ReleaseMatrix(S);
        gpu_sim.ReleaseMatrix(prod);
        S = new_S;
      }
      gpu_sim.ReleaseMatrix(q_col);
      gpu_sim.ReleaseMatrix(k_col);
    }

    // Softmax
    Matrix *P = nullptr;
    for (size_t r = 0; r < i + 1; ++r) {
      Matrix *row = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(S, r, row, kInSharedMemory);
      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row");
      gpu_sim.MatExp(row, exp_row);
      Matrix *sum_val = matrix_memory_allocator.Allocate("sum_val");
      gpu_sim.Sum(exp_row, sum_val);
      Matrix *soft_row = matrix_memory_allocator.Allocate("soft_row");
      gpu_sim.MatDiv(exp_row, sum_val, soft_row);
      if (P == nullptr) {
        P = soft_row;
      } else {
        Matrix *new_P = matrix_memory_allocator.Allocate("P_concat");
        gpu_sim.Concat(P, soft_row, new_P, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(P);
        gpu_sim.ReleaseMatrix(soft_row);
        P = new_P;
      }
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_val);
    }
    gpu_sim.ReleaseMatrix(S);

    // O = P * V
    Matrix *O = nullptr;
    for (size_t j = 0; j < i + 1; ++j) {
      Matrix *p_col = matrix_memory_allocator.Allocate("p_col");
      gpu_sim.GetColumn(P, j, p_col, kInSharedMemory);
      Matrix *v_row = matrix_memory_allocator.Allocate("v_row");
      gpu_sim.GetRow(acc_V, j, v_row, kInSharedMemory);
      Matrix *prod = matrix_memory_allocator.Allocate("prod_O");
      gpu_sim.MatMul(p_col, v_row, prod);
      if (O == nullptr) {
        O = prod;
      } else {
        Matrix *new_O = matrix_memory_allocator.Allocate("O_acc");
        gpu_sim.MatAdd(O, prod, new_O);
        gpu_sim.ReleaseMatrix(O);
        gpu_sim.ReleaseMatrix(prod);
        O = new_O;
      }
      gpu_sim.ReleaseMatrix(p_col);
      gpu_sim.ReleaseMatrix(v_row);
    }
    gpu_sim.ReleaseMatrix(P);
    gpu_sim.ReleaseMatrix(current_query);

    gpu_sim.MoveMatrixToGpuHbm(O);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*O);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
