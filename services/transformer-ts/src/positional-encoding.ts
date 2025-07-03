/**
 * Positional Encoding for Transformer
 */

export class PositionalEncoding {
  private encodings: number[][];
  
  constructor(maxLength: number, dModel: number) {
    this.encodings = this.generateEncodings(maxLength, dModel);
  }
  
  private generateEncodings(maxLength: number, dModel: number): number[][] {
    const encodings: number[][] = [];
    
    for (let pos = 0; pos < maxLength; pos++) {
      const encoding: number[] = [];
      for (let i = 0; i < dModel; i++) {
        if (i % 2 === 0) {
          encoding.push(Math.sin(pos / Math.pow(10000, i / dModel)));
        } else {
          encoding.push(Math.cos(pos / Math.pow(10000, (i - 1) / dModel)));
        }
      }
      encodings.push(encoding);
    }
    
    return encodings;
  }
  
  public addPositionalEncoding(embeddings: number[][]): number[][] {
    return embeddings.map((embedding, pos) => {
      if (pos >= this.encodings.length) {
        throw new Error(`Position ${pos} exceeds maximum length ${this.encodings.length}`);
      }
      return embedding.map((val, i) => val + this.encodings[pos][i]);
    });
  }
  
  public getEncoding(position: number): number[] {
    if (position >= this.encodings.length) {
      throw new Error(`Position ${position} exceeds maximum length ${this.encodings.length}`);
    }
    return [...this.encodings[position]];
  }
}
