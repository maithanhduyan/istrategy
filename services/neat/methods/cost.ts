// Converted from js/methods/cost.js to TypeScript
/**
 * Cost (loss) functions for neural networks
 */
const cost = {
    CROSS_ENTROPY: (target: number[], output: number[]): number => {
        let error = 0;
        for (let i = 0; i < output.length; i++) {
            error -= target[i] * Math.log(Math.max(output[i], 1e-15)) + (1 - target[i]) * Math.log(1 - Math.max(output[i], 1e-15));
        }
        return error / output.length;
    },
    MSE: (target: number[], output: number[]): number => {
        let error = 0;
        for (let i = 0; i < output.length; i++) {
            error += Math.pow(target[i] - output[i], 2);
        }
        return error / output.length;
    },
    BINARY: (target: number[], output: number[]): number => {
        let misses = 0;
        for (let i = 0; i < output.length; i++) {
            misses += (Math.round(target[i] * 2) !== Math.round(output[i] * 2)) ? 1 : 0;
        }
        return misses;
    },
    MAE: (target: number[], output: number[]): number => {
        let error = 0;
        for (let i = 0; i < output.length; i++) {
            error += Math.abs(target[i] - output[i]);
        }
        return error / output.length;
    },
    MAPE: (target: number[], output: number[]): number => {
        let error = 0;
        for (let i = 0; i < output.length; i++) {
            error += Math.abs((output[i] - target[i]) / Math.max(target[i], 1e-15));
        }
        return error / output.length;
    },
    MSLE: (target: number[], output: number[]): number => {
        let error = 0;
        for (let i = 0; i < output.length; i++) {
            error += Math.log(Math.max(target[i], 1e-15)) - Math.log(Math.max(output[i], 1e-15));
        }
        return error;
    }
};

export default cost;
