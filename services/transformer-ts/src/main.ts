/**
 * Main entry point and demo for Transformer model
 */

import { TransformerModel } from './transformer-model.js';
import { relu, tanh, sigmoid } from './activation.js';

// Demo configuration
const VOCAB_SIZE = 1000;
const D_MODEL = 512;
const NUM_HEADS = 8;
const NUM_LAYERS = 6;
const D_FF = 2048;
const MAX_LENGTH = 100;

// Special tokens
const START_TOKEN = 0;
const END_TOKEN = 1;
const PAD_TOKEN = 2;

function demonstrateTransformer() {
  console.log('=== Transformer Model Demo ===');
  
  // Initialize model
  const model = new TransformerModel(
    VOCAB_SIZE,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    D_FF,
    MAX_LENGTH,
    0.1,
    relu
  );
  
  console.log(`Model initialized with ${model.getParameterCount().toLocaleString()} parameters`);
  
  // Demo input sequence (e.g., "Hello world")
  const inputTokens = [4, 5, 6, 7, 8]; // Dummy tokens
  const outputTokens = [START_TOKEN, 9, 10, 11]; // Dummy target sequence
  
  console.log('\\nInput tokens:', inputTokens);
  console.log('Output tokens:', outputTokens);
  
  // Encoder forward pass
  console.log('\\n--- Encoder Forward Pass ---');
  const encoderOutput = model.encode(inputTokens);
  console.log(`Encoder output shape: [${encoderOutput.length}, ${encoderOutput[0].length}]`);
  console.log('First encoder output vector (first 5 values):', 
    encoderOutput[0].slice(0, 5).map(x => x.toFixed(4)));
  
  // Decoder forward pass
  console.log('\\n--- Decoder Forward Pass ---');
  const decoderOutput = model.decode(outputTokens, encoderOutput);
  console.log(`Decoder output shape: [${decoderOutput.length}, ${decoderOutput[0].length}]`);
  console.log('First decoder output vector (first 5 values):', 
    decoderOutput[0].slice(0, 5).map(x => x.toFixed(4)));
  
  // Full forward pass
  console.log('\\n--- Full Forward Pass ---');
  const logits = model.forward(inputTokens, outputTokens);
  console.log(`Final logits shape: [${logits.length}, ${logits[0].length}]`);
  console.log('First logits vector (first 5 values):', 
    logits[0].slice(0, 5).map(x => x.toFixed(4)));
  
  // Generate next token probabilities
  console.log('\\n--- Next Token Generation ---');
  const nextTokenProbs = model.generateNextToken(inputTokens, outputTokens);
  console.log('Next token probabilities (first 10 tokens):', 
    nextTokenProbs.slice(0, 10).map(x => x.toFixed(6)));
  
  // Generate sequence
  console.log('\\n--- Sequence Generation ---');
  const generatedSequence = model.generateSequence(
    inputTokens, 
    START_TOKEN, 
    END_TOKEN, 
    20
  );
  console.log('Generated sequence:', generatedSequence);
  
  console.log('\\n=== Demo Complete ===');
}

// Test different components
function testComponents() {
  console.log('\\n=== Component Tests ===');
  
  // Test with smaller model for faster execution
  const testModel = new TransformerModel(
    100,   // smaller vocab
    128,   // smaller d_model
    4,     // fewer heads
    2,     // fewer layers
    512,   // smaller d_ff
    50,    // shorter max length
    0.0,   // no dropout for testing
    tanh   // different activation
  );
  
  console.log(`Test model parameters: ${testModel.getParameterCount().toLocaleString()}`);
  
  // Test with simple sequences
  const simpleInput = [3, 4, 5];
  const simpleOutput = [START_TOKEN, 6, 7];
  
  try {
    const result = testModel.forward(simpleInput, simpleOutput);
    console.log('✅ Forward pass successful');
    console.log(`Result shape: [${result.length}, ${result[0].length}]`);
    
    const generated = testModel.generateSequence(simpleInput, START_TOKEN, END_TOKEN, 10);
    console.log('✅ Generation successful');
    console.log('Generated tokens:', generated);
    
  } catch (error) {
    console.error('❌ Test failed:', error instanceof Error ? error.message : String(error));
  }
}

// Performance benchmark
function performanceBenchmark() {
  console.log('\\n=== Performance Benchmark ===');
  
  const benchmarkModel = new TransformerModel(
    50,    // very small vocab
    64,    // very small d_model
    2,     // minimal heads
    1,     // single layer
    128,   // small d_ff
    20,    // short sequences
    0.0,   // no dropout
    relu
  );
  
  const inputSeq = [1, 2, 3, 4, 5];
  const outputSeq = [START_TOKEN, 6, 7, 8];
  
  console.log('Running 10 forward passes...');
  const startTime = performance.now();
  
  for (let i = 0; i < 10; i++) {
    benchmarkModel.forward(inputSeq, outputSeq);
  }
  
  const endTime = performance.now();
  const avgTime = (endTime - startTime) / 10;
  
  console.log(`Average forward pass time: ${avgTime.toFixed(2)}ms`);
  console.log(`Model parameters: ${benchmarkModel.getParameterCount().toLocaleString()}`);
}

// Main execution
function main() {
  try {
    demonstrateTransformer();
    testComponents();
    performanceBenchmark();
  } catch (error) {
    console.error('Error in main execution:', error);
  }
}

// Export for use as module
export { TransformerModel, main };

// Run if this is the main module
if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  main();
}
