/**
 * Layer Normalization for Transformer
 */

import { layerNorm, addVectors } from './math-utils.js';

export class LayerNormalization {
  private dModel: number;
  private gamma: number[]; // Scale parameter
  private beta: number[]; // Shift parameter
  private epsilon: number;
  
  constructor(dModel: number, epsilon: number = 1e-6) {
    this.dModel = dModel;
    this.epsilon = epsilon;
    
    // Initialize parameters
    this.gamma = new Array(dModel).fill(1.0);
    this.beta = new Array(dModel).fill(0.0);
  }
  
  public forward(x: number[][]): number[][] {
    return x.map(vector => this.forwardSingle(vector));
  }
  
  private forwardSingle(x: number[]): number[] {
    // Apply layer normalization
    const normalized = layerNorm(x, this.epsilon);
    
    // Apply scale and shift
    return normalized.map((val, i) => val * this.gamma[i] + this.beta[i]);
  }
  
  public getParameters(): { gamma: number[]; beta: number[] } {
    return {
      gamma: [...this.gamma],
      beta: [...this.beta]
    };
  }
}
