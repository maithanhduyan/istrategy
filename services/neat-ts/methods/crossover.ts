// Converted from js/methods/crossover.js to TypeScript
/**
 * Crossover methods for genetic algorithms
 */
const crossover = {
    SINGLE_POINT: {
        name: 'SINGLE_POINT',
        config: [0.4]
    },
    TWO_POINT: {
        name: 'TWO_POINT',
        config: [0.4, 0.9]
    },
    UNIFORM: {
        name: 'UNIFORM'
    },
    AVERAGE: {
        name: 'AVERAGE'
    }
};

export default crossover;
