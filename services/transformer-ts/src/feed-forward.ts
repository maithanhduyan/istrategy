/**
 * Feed Forward Network for Transformer
 */

import { vectorMatMul, addVectors, initializeWeights } from './math-utils.js';
import { ActivationFn, relu } from './activation.js';

export class FeedForwardNetwork {
  private dModel: number;
  private dFF: number;
  private w1: number[][]; // First linear layer weights
  private b1: number[]; // First layer bias
  private w2: number[][]; // Second linear layer weights
  private b2: number[]; // Second layer bias
  private activation: ActivationFn;
  
  constructor(dModel: number, dFF: number, activation: ActivationFn = relu) {
    this.dModel = dModel;
    this.dFF = dFF;
    this.activation = activation;
    
    // Initialize weights and biases
    this.w1 = initializeWeights(dFF, dModel);
    this.b1 = new Array(dFF).fill(0);
    this.w2 = initializeWeights(dModel, dFF);
    this.b2 = new Array(dModel).fill(0);
  }
  
  public forward(x: number[][]): number[][] {
    return x.map(vector => this.forwardSingle(vector));
  }
  
  private forwardSingle(x: number[]): number[] {
    // First linear transformation + bias + activation
    const h1 = addVectors(this.matVecMul(this.w1, x), this.b1);
    const activated = h1.map(this.activation);
    
    // Second linear transformation + bias
    return addVectors(this.matVecMul(this.w2, activated), this.b2);
  }
  
  private matVecMul(matrix: number[][], vector: number[]): number[] {
    if (matrix[0].length !== vector.length) {
      throw new Error(`Matrix-vector dimensions incompatible: ${matrix[0].length} vs ${vector.length}`);
    }
    return matrix.map(row => vector.reduce((sum, val, i) => sum + val * row[i], 0));
  }
  
  public getParameters(): {
    w1: number[][];
    b1: number[];
    w2: number[][];
    b2: number[];
  } {
    return {
      w1: this.w1.map(row => [...row]),
      b1: [...this.b1],
      w2: this.w2.map(row => [...row]),
      b2: [...this.b2]
    };
  }
}
