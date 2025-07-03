/**
 * Transformer Encoder Layer
 */

import { MultiHeadAttention } from './multi-head-attention.js';
import { FeedForwardNetwork } from './feed-forward.js';
import { LayerNormalization } from './layer-normalization.js';
import { addVectors } from './math-utils.js';
import { ActivationFn, relu } from './activation.js';

export class TransformerEncoderLayer {
  private multiHeadAttention: MultiHeadAttention;
  private feedForward: FeedForwardNetwork;
  private layerNorm1: LayerNormalization;
  private layerNorm2: LayerNormalization;
  private dropoutRate: number;
  
  constructor(
    dModel: number,
    numHeads: number,
    dFF: number,
    dropoutRate: number = 0.1,
    activation: ActivationFn = relu
  ) {
    this.multiHeadAttention = new MultiHeadAttention(dModel, numHeads);
    this.feedForward = new FeedForwardNetwork(dModel, dFF, activation);
    this.layerNorm1 = new LayerNormalization(dModel);
    this.layerNorm2 = new LayerNormalization(dModel);
    this.dropoutRate = dropoutRate;
  }
  
  public forward(x: number[][], mask?: number[][]): number[][] {
    // Self-attention with residual connection and layer norm
    const attentionOutput = this.multiHeadAttention.forward(x, x, x, mask);
    const residual1 = this.addResidualConnection(x, attentionOutput);
    const normed1 = this.layerNorm1.forward(residual1);
    
    // Feed-forward with residual connection and layer norm
    const ffOutput = this.feedForward.forward(normed1);
    const residual2 = this.addResidualConnection(normed1, ffOutput);
    const normed2 = this.layerNorm2.forward(residual2);
    
    return normed2;
  }
  
  private addResidualConnection(input: number[][], output: number[][]): number[][] {
    return input.map((inputVec, i) => addVectors(inputVec, output[i]));
  }
  
  // Simple dropout implementation (for training)
  private applyDropout(x: number[][], rate: number): number[][] {
    if (rate === 0) return x;
    
    return x.map(vector => 
      vector.map(val => Math.random() > rate ? val / (1 - rate) : 0)
    );
  }
}
