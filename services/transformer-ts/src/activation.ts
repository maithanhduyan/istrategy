// === activation.ts ===
export type ActivationFn = (x: number) => number;

export const relu: ActivationFn = x => Math.max(0, x);
export const tanh: ActivationFn = x => Math.tanh(x);
export const sigmoid: ActivationFn = x => 1 / (1 + Math.exp(-x));