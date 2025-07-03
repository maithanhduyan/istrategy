/**
 * Transformer Decoder Layer
 */

import { MultiHeadAttention } from './multi-head-attention.js';
import { FeedForwardNetwork } from './feed-forward.js';
import { LayerNormalization } from './layer-normalization.js';
import { addVectors } from './math-utils.js';
import { ActivationFn, relu } from './activation.js';

export class TransformerDecoderLayer {
  private maskedMultiHeadAttention: MultiHeadAttention;
  private crossAttention: MultiHeadAttention;
  private feedForward: FeedForwardNetwork;
  private layerNorm1: LayerNormalization;
  private layerNorm2: LayerNormalization;
  private layerNorm3: LayerNormalization;
  private dropoutRate: number;
  
  constructor(
    dModel: number,
    numHeads: number,
    dFF: number,
    dropoutRate: number = 0.1,
    activation: ActivationFn = relu
  ) {
    this.maskedMultiHeadAttention = new MultiHeadAttention(dModel, numHeads);
    this.crossAttention = new MultiHeadAttention(dModel, numHeads);
    this.feedForward = new FeedForwardNetwork(dModel, dFF, activation);
    this.layerNorm1 = new LayerNormalization(dModel);
    this.layerNorm2 = new LayerNormalization(dModel);
    this.layerNorm3 = new LayerNormalization(dModel);
    this.dropoutRate = dropoutRate;
  }
  
  public forward(
    x: number[][], 
    encoderOutput: number[][], 
    causalMask?: number[][], 
    paddingMask?: number[][]
  ): number[][] {
    // Masked self-attention with residual connection and layer norm
    const maskedAttentionOutput = this.maskedMultiHeadAttention.forward(
      x, x, x, causalMask
    );
    const residual1 = this.addResidualConnection(x, maskedAttentionOutput);
    const normed1 = this.layerNorm1.forward(residual1);
    
    // Cross-attention with encoder output
    const crossAttentionOutput = this.crossAttention.forward(
      normed1, encoderOutput, encoderOutput, paddingMask
    );
    const residual2 = this.addResidualConnection(normed1, crossAttentionOutput);
    const normed2 = this.layerNorm2.forward(residual2);
    
    // Feed-forward with residual connection and layer norm
    const ffOutput = this.feedForward.forward(normed2);
    const residual3 = this.addResidualConnection(normed2, ffOutput);
    const normed3 = this.layerNorm3.forward(residual3);
    
    return normed3;
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
