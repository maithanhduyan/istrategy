/**
 * Complete Transformer Model
 */

import { TransformerEncoderLayer } from './transformer-encoder-layer.js';
import { TransformerDecoderLayer } from './transformer-decoder-layer.js';
import { PositionalEncoding } from './positional-encoding.js';
import { MultiHeadAttention } from './multi-head-attention.js';
import { vectorMatMul, initializeWeights, softmax } from './math-utils.js';
import { ActivationFn, relu } from './activation.js';

export class TransformerModel {
  private dModel: number;
  private numHeads: number;
  private numLayers: number;
  private dFF: number;
  private maxLength: number;
  private vocabSize: number;
  
  private inputEmbedding: number[][]; // [vocabSize, dModel]
  private outputEmbedding: number[][]; // [vocabSize, dModel]
  private positionalEncoding: PositionalEncoding;
  private encoderLayers: TransformerEncoderLayer[];
  private decoderLayers: TransformerDecoderLayer[];
  private outputProjection: number[][]; // [vocabSize, dModel]
  
  constructor(
    vocabSize: number,
    dModel: number = 512,
    numHeads: number = 8,
    numLayers: number = 6,
    dFF: number = 2048,
    maxLength: number = 1000,
    dropoutRate: number = 0.1,
    activation: ActivationFn = relu
  ) {
    this.vocabSize = vocabSize;
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.numLayers = numLayers;
    this.dFF = dFF;
    this.maxLength = maxLength;
    
    // Initialize embeddings
    this.inputEmbedding = initializeWeights(vocabSize, dModel, 0.1);
    this.outputEmbedding = initializeWeights(vocabSize, dModel, 0.1);
    
    // Initialize positional encoding
    this.positionalEncoding = new PositionalEncoding(maxLength, dModel);
    
    // Initialize encoder layers
    this.encoderLayers = [];
    for (let i = 0; i < numLayers; i++) {
      this.encoderLayers.push(
        new TransformerEncoderLayer(dModel, numHeads, dFF, dropoutRate, activation)
      );
    }
    
    // Initialize decoder layers
    this.decoderLayers = [];
    for (let i = 0; i < numLayers; i++) {
      this.decoderLayers.push(
        new TransformerDecoderLayer(dModel, numHeads, dFF, dropoutRate, activation)
      );
    }
    
    // Initialize output projection
    this.outputProjection = initializeWeights(vocabSize, dModel, 0.1);
  }
  
  // Encode input sequence
  public encode(inputTokens: number[]): number[][] {
    // Token embedding
    const embedded = this.embedTokens(inputTokens, this.inputEmbedding);
    
    // Add positional encoding
    const withPositions = this.positionalEncoding.addPositionalEncoding(embedded);
    
    // Pass through encoder layers
    let encoderOutput = withPositions;
    for (const layer of this.encoderLayers) {
      encoderOutput = layer.forward(encoderOutput);
    }
    
    return encoderOutput;
  }
  
  // Decode output sequence
  public decode(
    outputTokens: number[], 
    encoderOutput: number[][], 
    causalMask?: number[][]
  ): number[][] {
    // Token embedding
    const embedded = this.embedTokens(outputTokens, this.outputEmbedding);
    
    // Add positional encoding
    const withPositions = this.positionalEncoding.addPositionalEncoding(embedded);
    
    // Create causal mask if not provided
    const mask = causalMask || MultiHeadAttention.createCausalMask(outputTokens.length);
    
    // Pass through decoder layers
    let decoderOutput = withPositions;
    for (const layer of this.decoderLayers) {
      decoderOutput = layer.forward(decoderOutput, encoderOutput, mask);
    }
    
    return decoderOutput;
  }
  
  // Full forward pass
  public forward(inputTokens: number[], outputTokens: number[]): number[][] {
    const encoderOutput = this.encode(inputTokens);
    const decoderOutput = this.decode(outputTokens, encoderOutput);
    
    // Project to vocabulary
    return this.projectToVocab(decoderOutput);
  }
  
  // Generate next token probabilities
  public generateNextToken(inputTokens: number[], outputTokens: number[]): number[] {
    const logits = this.forward(inputTokens, outputTokens);
    const lastLogits = logits[logits.length - 1];
    return softmax(lastLogits);
  }
  
  // Greedy decoding
  public generateSequence(
    inputTokens: number[], 
    startToken: number, 
    endToken: number, 
    maxLength: number = 50
  ): number[] {
    const generated: number[] = [startToken];
    
    for (let i = 0; i < maxLength; i++) {
      const probs = this.generateNextToken(inputTokens, generated);
      const nextToken = this.argmax(probs);
      
      generated.push(nextToken);
      
      if (nextToken === endToken) {
        break;
      }
    }
    
    return generated;
  }
  
  private embedTokens(tokens: number[], embedding: number[][]): number[][] {
    return tokens.map(token => {
      if (token >= this.vocabSize) {
        throw new Error(`Token ${token} exceeds vocabulary size ${this.vocabSize}`);
      }
      return [...embedding[token]];
    });
  }
  
  private projectToVocab(hiddenStates: number[][]): number[][] {
    return hiddenStates.map(state => this.matVecMul(this.outputProjection, state));
  }
  
  private matVecMul(matrix: number[][], vector: number[]): number[] {
    if (matrix[0].length !== vector.length) {
      throw new Error(`Matrix-vector dimensions incompatible: ${matrix[0].length} vs ${vector.length}`);
    }
    return matrix.map(row => vector.reduce((sum, val, i) => sum + val * row[i], 0));
  }
  
  private argmax(arr: number[]): number {
    return arr.indexOf(Math.max(...arr));
  }
  
  // Get model parameters count
  public getParameterCount(): number {
    let count = 0;
    
    // Embeddings
    count += this.vocabSize * this.dModel * 2; // input + output embeddings
    count += this.vocabSize * this.dModel; // output projection
    
    // Encoder layers
    count += this.numLayers * this.getEncoderLayerParams();
    
    // Decoder layers  
    count += this.numLayers * this.getDecoderLayerParams();
    
    return count;
  }
  
  private getEncoderLayerParams(): number {
    // Multi-head attention: 4 * dModel * dModel (Q, K, V, O)
    // Feed-forward: 2 * dModel * dFF + dModel + dFF (weights + biases)
    // Layer norms: 2 * 2 * dModel (gamma + beta for each norm)
    return 4 * this.dModel * this.dModel + 
           2 * this.dModel * this.dFF + this.dModel + this.dFF +
           4 * this.dModel;
  }
  
  private getDecoderLayerParams(): number {
    // 2 Multi-head attentions: 2 * 4 * dModel * dModel
    // Feed-forward: 2 * dModel * dFF + dModel + dFF
    // Layer norms: 3 * 2 * dModel (gamma + beta for each norm)
    return 8 * this.dModel * this.dModel + 
           2 * this.dModel * this.dFF + this.dModel + this.dFF +
           6 * this.dModel;
  }
}
