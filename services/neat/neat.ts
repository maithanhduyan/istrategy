// Converted from js/neat.js to TypeScript
import methods from './methods/methods';

let selection = methods.selection;

class Neat {
    input: number;
    output: number;
    fitness: Function;
    equal: boolean;
    clear: boolean;
    popsize: number;
    elitism: number;
    provenance: number;
    mutationRate: number;
    mutationAmount: number;
    fitnessPopulation: boolean;
    selection: any;
    crossover: any;
    mutation: any;
    template: any;
    maxNodes: number;
    maxConns: number;
    maxGates: number;
    selectMutationMethod: Function;
    generation: number;
    population: any[] = [];

    constructor(input: number, output: number, fitness: Function, options: any) {
        this.input = input;
        this.output = output;
        this.fitness = fitness;
        options = options || {};
        this.equal = options.equal || false;
        this.clear = options.clear || false;
        this.popsize = options.popsize || 50;
        this.elitism = options.elitism || 0;
        this.provenance = options.provenance || 0;
        this.mutationRate = options.mutationRate || 0.3;
        this.mutationAmount = options.mutationAmount || 1;
        this.fitnessPopulation = options.fitnessPopulation || false;
        this.selection = options.selection || methods.selection.POWER;
        this.crossover = options.crossover || [
            methods.crossover.SINGLE_POINT,
            methods.crossover.TWO_POINT,
            methods.crossover.UNIFORM,
            methods.crossover.AVERAGE
        ];
        this.mutation = options.mutation || methods.mutation.MOD_ACTIVATION;
        this.template = options.network || false;
        this.maxNodes = options.maxNodes || Infinity;
        this.maxConns = options.maxConns || Infinity;
        this.maxGates = options.maxGates || Infinity;
        // Assign default selectMutationMethod if not provided
        this.selectMutationMethod = typeof options.mutationSelection === 'function' ? options.mutationSelection.bind(this) : () => methods.mutation.MOD_ACTIVATION;
        this.generation = 0;
        // Call createPool if defined
        if (typeof (this as any).createPool === 'function') {
            (this as any).createPool(this.template);
        }
    }
    // ...existing methods from js/neat.js...
}

export default Neat;
