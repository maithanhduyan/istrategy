// Converted from js/methods/rate.js to TypeScript
/**
 * Learning rate policies for neural networks
 */
const rate = {
    FIXED: function () {
        let func = function (baseRate: number) {
            return baseRate;
        };
        return func;
    },
    STEP: function (gamma?: number, stepSize?: number) {
        gamma = gamma || 0.9;
        stepSize = stepSize || 100;
        let func = function (baseRate: number, iteration: number) {
            return baseRate * Math.pow(gamma as number, Math.floor(iteration / (stepSize as number)));
        };
        return func;
    },
    EXP: function (gamma?: number) {
        gamma = gamma || 0.999;
        let func = function (baseRate: number, iteration: number) {
            return baseRate * Math.pow(gamma as number, iteration);
        };
        return func;
    },
    INV: function (gamma?: number, power?: number) {
        gamma = gamma || 0.001;
        power = power || 2;
        let func = function (baseRate: number, iteration: number) {
            return baseRate * Math.pow(1 + (gamma as number) * iteration, -(power as number));
        };
        return func;
    }
};

export default rate;
