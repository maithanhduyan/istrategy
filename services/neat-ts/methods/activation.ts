// Converted from js/methods/activation.js to TypeScript
/**
 * Activation functions for neural networks
 */
const activation = {
    LOGISTIC: (x: number, derivate?: boolean): number => {
        let fx = 1 / (1 + Math.exp(-x));
        if (!derivate) return fx;
        return fx * (1 - fx);
    },
    TANH: (x: number, derivate?: boolean): number => {
        if (derivate) return 1 - Math.pow(Math.tanh(x), 2);
        return Math.tanh(x);
    },
    IDENTITY: (x: number, derivate?: boolean): number => {
        return derivate ? 1 : x;
    },
    STEP: (x: number, derivate?: boolean): number => {
        return derivate ? 0 : x > 0 ? 1 : 0;
    },
    RELU: (x: number, derivate?: boolean): number => {
        if (derivate) return x > 0 ? 1 : 0;
        return x > 0 ? x : 0;
    },
    SOFTSIGN: (x: number, derivate?: boolean): number => {
        let d = 1 + Math.abs(x);
        if (derivate) return x / Math.pow(d, 2);
        return x / d;
    },
    SINUSOID: (x: number, derivate?: boolean): number => {
        if (derivate) return Math.cos(x);
        return Math.sin(x);
    },
    GAUSSIAN: (x: number, derivate?: boolean): number => {
        let d = Math.exp(-Math.pow(x, 2));
        if (derivate) return -2 * x * d;
        return d;
    },
    BENT_IDENTITY: (x: number, derivate?: boolean): number => {
        let d = Math.sqrt(Math.pow(x, 2) + 1);
        if (derivate) return x / (2 * d) + 1;
        return (d - 1) / 2 + x;
    },
    BIPOLAR: (x: number, derivate?: boolean): number => {
        return derivate ? 0 : x > 0 ? 1 : -1;
    },
    BIPOLAR_SIGMOID: (x: number, derivate?: boolean): number => {
        let d = 2 / (1 + Math.exp(-x)) - 1;
        if (derivate) return 1 / 2 * (1 + d) * (1 - d);
        return d;
    },
    HARD_TANH: (x: number, derivate?: boolean): number => {
        if (derivate) return x > -1 && x < 1 ? 1 : 0;
        return Math.max(-1, Math.min(1, x));
    },
    ABSOLUTE: (x: number, derivate?: boolean): number => {
        if (derivate) return x < 0 ? -1 : 1;
        return Math.abs(x);
    },
    INVERSE: (x: number, derivate?: boolean): number => {
        if (derivate) return -1;
        return 1 - x;
    },
    SELU: (x: number, derivate?: boolean): number => {
        const alpha = 1.6732632423543772;
        const scale = 1.0507009873554805;
        let fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
        if (derivate) { return x > 0 ? scale : (fx + alpha) * scale; }
        return fx * scale;
    }
};

export default activation;
