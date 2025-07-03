/**
 * Multi-Head Attention mechanism for Transformer
 */

import { dot, softmax, vectorMatMul, initializeWeights } from './math-utils.js';

export class MultiHeadAttention {
  private dModel: number;
  private numHeads: number;
  private dK: number;
  private dV: number;
  private wQ: number[][][]; // Query weights [numHeads][dModel][dK]
  private wK: number[][][]; // Key weights [numHeads][dModel][dK]
  private wV: number[][][]; // Value weights [numHeads][dModel][dV]
  private wO: number[][]; // Output projection weights [dModel][dModel]
  
  constructor(dModel: number, numHeads: number) {
    if (dModel % numHeads !== 0) {
      throw new Error(`dModel (${dModel}) must be divisible by numHeads (${numHeads})`);
    }
    
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.dK = dModel / numHeads;
    this.dV = dModel / numHeads;
    
    // Initialize weight matrices
    this.wQ = this.initializeHeadWeights(numHeads, dModel, this.dK);
    this.wK = this.initializeHeadWeights(numHeads, dModel, this.dK);
    this.wV = this.initializeHeadWeights(numHeads, dModel, this.dV);
    this.wO = initializeWeights(dModel, dModel);
  }
  
  private initializeHeadWeights(numHeads: number, dModel: number, dOut: number): number[][][] {
    return Array.from({ length: numHeads }, () =>
      initializeWeights(dOut, dModel) // Đổi thứ tự: dOut x dModel
    );
  }
  
  private scaledDotProductAttention(
    Q: number[][], 
    K: number[][], 
    V: number[][], 
    mask?: number[][]
  ): number[][] {
    if (!Q || !K || !V || Q.length === 0 || K.length === 0 || V.length === 0) {
      throw new Error('Invalid input to scaledDotProductAttention');
    }
    
    const seqLenQ = Q.length;
    const seqLenK = K.length;
    const scores: number[][] = [];
    
    // Calculate attention scores: Q * K^T
    for (let i = 0; i < seqLenQ; i++) {
      const scoreRow: number[] = [];
      for (let j = 0; j < seqLenK; j++) {
        if (!Q[i] || !K[j]) {
          throw new Error(`Q[${i}] or K[${j}] is undefined`);
        }
        scoreRow.push(dot(Q[i], K[j]));
      }
      scores.push(scoreRow);
    }
    
    // Scale by sqrt(dK)
    const scaledScores = scores.map(row => 
      row.map(score => score / Math.sqrt(this.dK))
    );
    
    // Apply mask if provided (for decoder self-attention)
    if (mask) {
      for (let i = 0; i < seqLenQ; i++) {
        for (let j = 0; j < seqLenK; j++) {
          if (mask[i] && mask[i][j] === 0) {
            scaledScores[i][j] = -Infinity;
          }
        }
      }
    }
    
    // Apply softmax to get attention weights
    const attentionWeights = scaledScores.map(row => softmax(row));
    
    // Apply attention weights to values
    const output: number[][] = [];
    for (let i = 0; i < seqLenQ; i++) {
      const outputRow = new Array(this.dV).fill(0);
      for (let j = 0; j < seqLenK; j++) {
        for (let k = 0; k < this.dV; k++) {
          outputRow[k] += attentionWeights[i][j] * V[j][k];
        }
      }
      output.push(outputRow);
    }
    
    return output;
  }
  
  public forward(
    query: number[][], 
    key: number[][], 
    value: number[][], 
    mask?: number[][]
  ): number[][] {
    const seqLen = query.length;
    const headOutputs: number[][][] = [];
    
    // Process each attention head
    for (let h = 0; h < this.numHeads; h++) {
      // Linear transformations for Q, K, V
      const Q = query.map(q => {
        if (!q || !this.wQ[h]) {
          throw new Error(`Invalid query or weight at head ${h}`);
        }
        return this.matVecMul(this.wQ[h], q);
      });
      const K = key.map(k => {
        if (!k || !this.wK[h]) {
          throw new Error(`Invalid key or weight at head ${h}`);
        }
        return this.matVecMul(this.wK[h], k);
      });
      const V = value.map(v => {
        if (!v || !this.wV[h]) {
          throw new Error(`Invalid value or weight at head ${h}`);
        }
        return this.matVecMul(this.wV[h], v);
      });
      
      // Scaled dot-product attention
      const headOutput = this.scaledDotProductAttention(Q, K, V, mask);
      headOutputs.push(headOutput);
    }
    
    // Concatenate all head outputs
    const concatenated: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
      const concatRow: number[] = [];
      for (let h = 0; h < this.numHeads; h++) {
        concatRow.push(...headOutputs[h][i]);
      }
      concatenated.push(concatRow);
    }
    
    // Apply output projection
    return concatenated.map(row => this.matVecMul(this.wO, row));
  }
  
  // Create causal mask for decoder self-attention
  public static createCausalMask(seqLen: number): number[][] {
    const mask: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
      const row: number[] = [];
      for (let j = 0; j < seqLen; j++) {
        row.push(j <= i ? 1 : 0);
      }
      mask.push(row);
    }
    return mask;
  }

  private matVecMul(matrix: number[][], vector: number[]): number[] {
    if (matrix[0].length !== vector.length) {
      throw new Error(`Matrix-vector dimensions incompatible: ${matrix[0].length} vs ${vector.length}`);
    }
    return matrix.map(row => dot(row, vector));
  }
}
