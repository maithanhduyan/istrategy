/**
 * Mathematical utilities for Transformer model
 */

export function dot(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vector dimensions must match: ${a.length} vs ${b.length}`);
  }
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

export function softmax(x: number[]): number[] {
  const max = Math.max(...x);
  const exps = x.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

export function matMul(a: number[][], b: number[][]): number[][] {
  if (a[0].length !== b.length) {
    throw new Error(`Matrix dimensions incompatible: ${a[0].length} vs ${b.length}`);
  }
  return a.map(row => b[0].map((_, colIndex) => 
    row.reduce((sum, val, rowIndex) => sum + val * b[rowIndex][colIndex], 0)
  ));
}

export function vectorMatMul(matrix: number[][], vector: number[]): number[] {
  if (matrix[0].length !== vector.length) {
    throw new Error(`Dimensions incompatible: ${matrix[0].length} vs ${vector.length}`);
  }
  return matrix.map(row => dot(row, vector));
}

export function addVectors(a: number[], b: number[]): number[] {
  if (a.length !== b.length) {
    throw new Error(`Vector dimensions must match: ${a.length} vs ${b.length}`);
  }
  return a.map((val, i) => val + b[i]);
}

export function layerNorm(x: number[], epsilon: number = 1e-6): number[] {
  const mean = x.reduce((sum, val) => sum + val, 0) / x.length;
  const variance = x.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / x.length;
  const std = Math.sqrt(variance + epsilon);
  return x.map(val => (val - mean) / std);
}

export function initializeWeights(rows: number, cols: number, scale: number = 0.1): number[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() - 0.5) * scale)
  );
}
