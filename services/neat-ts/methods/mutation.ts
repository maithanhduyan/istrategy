// Converted from js/methods/mutation.js to TypeScript
import activation from './activation';

/**
 * Mutation methods for genetic algorithms
 */
const mutation = {
    ADD_NODE: {
        name: 'ADD_NODE'
    },
    SUB_NODE: {
        name: 'SUB_NODE',
        keep_gates: true
    },
    ADD_CONN: {
        name: 'ADD_CONN'
    },
    SUB_CONN: {
        name: 'REMOVE_CONN'
    },
    MOD_WEIGHT: {
        name: 'MOD_WEIGHT',
        min: -1,
        max: 1
    },
    MOD_BIAS: {
        name: 'MOD_BIAS',
        min: -1,
        max: 1
    },
    MOD_ACTIVATION: {
        name: 'MOD_ACTIVATION',
        mutateOutput: true,
        allowed: [
            activation.LOGISTIC,
            activation.TANH,
            activation.RELU,
            activation.IDENTITY,
            activation.STEP,
            activation.SOFTSIGN,
            activation.SINUSOID,
            activation.GAUSSIAN,
            activation.BENT_IDENTITY,
            activation.BIPOLAR,
            activation.BIPOLAR_SIGMOID,
            activation.HARD_TANH,
            activation.ABSOLUTE,
            activation.INVERSE,
            activation.SELU
        ]
    },
    ADD_SELF_CONN: {
        name: 'ADD_SELF_CONN'
    },
    SUB_SELF_CONN: {
        name: 'SUB_SELF_CONN'
    },
    ADD_GATE: {
        name: 'ADD_GATE'
    },
    SUB_GATE: {
        name: 'SUB_GATE'
    },
    ADD_BACK_CONN: {
        name: 'ADD_BACK_CONN'
    },
    SUB_BACK_CONN: {
        name: 'SUB_BACK_CONN'
    },
    SWAP_NODES: {
        name: 'SWAP_NODES',
        mutateOutput: true
    }
};

export default mutation;
