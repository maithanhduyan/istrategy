// Converted from js/architecture/architect.js to TypeScript
import methods from '../methods/methods';
import Group from './group';
import Layer from './layer';
import Network from './network';
import Node from './node';

/**
 * Architect module for building various neural network architectures.
 */
const architect = {
    Construct: function (list: (Group | Layer | Node)[]): Network {
        let network = new Network(0, 0);
        let nodes: any[] = [];
        let i: number;
        for (i = 0; i < list.length; i++) {
            let j: number;
            if (list[i] instanceof Group) {
                for (j = 0; j < (list[i] as Group).nodes.length; j++) {
                    nodes.push((list[i] as Group).nodes[j]);
                }
            } else if (list[i] instanceof Layer) {
                for (j = 0; j < (list[i] as Layer).nodes.length; j++) {
                    for (let k = 0; k < ((list[i] as Layer).nodes[j] as Group).nodes.length; k++) {
                        nodes.push(((list[i] as Layer).nodes[j] as Group).nodes[k]);
                    }
                }
            } else if (list[i] instanceof Node) {
                nodes.push(list[i]);
            }
        }
        let inputs: any[] = [];
        let outputs: any[] = [];
        for (i = nodes.length - 1; i >= 0; i--) {
            if (nodes[i].type === 'output' || nodes[i].connections.out.length + nodes[i].connections.gated.length === 0) {
                nodes[i].type = 'output';
                network.output++;
                outputs.push(nodes[i]);
                nodes.splice(i, 1);
            } else if (nodes[i].type === 'input' || !nodes[i].connections.in.length) {
                nodes[i].type = 'input';
                network.input++;
                inputs.push(nodes[i]);
                nodes.splice(i, 1);
            }
        }
        nodes = inputs.concat(nodes).concat(outputs);
        if (network.input === 0 || network.output === 0) {
            throw new Error('Given nodes have no clear input/output node!');
        }
        for (i = 0; i < nodes.length; i++) {
            let j: number;
            for (j = 0; j < nodes[i].connections.out.length; j++) {
                network.connections.push(nodes[i].connections.out[j]);
            }
            for (j = 0; j < nodes[i].connections.gated.length; j++) {
                network.gates.push(nodes[i].connections.gated[j]);
            }
            if (nodes[i].connections.self.weight !== 0) {
                network.selfconns.push(nodes[i].connections.self);
            }
        }
        network.nodes = nodes;
        return network;
    },
    Perceptron: function (...layers: number[]): Network {
        if (layers.length < 3) {
            throw new Error('You have to specify at least 3 layers');
        }
        let nodes: Group[] = [];
        nodes.push(new Group(layers[0]));
        for (let i = 1; i < layers.length; i++) {
            let layer = new Group(layers[i]);
            nodes.push(layer);
            nodes[i - 1].connect(layer, methods.connection.ALL_TO_ALL);
        }
        return architect.Construct(nodes);
    },
    Random: function (input: number, hidden: number, output: number, options?: any): Network {
        options = options || {};
        let connections = options.connections || hidden * 2;
        let backconnections = options.backconnections || 0;
        let selfconnections = options.selfconnections || 0;
        let gates = options.gates || 0;
        let network = new Network(input, output);
        let i: number;
        for (i = 0; i < hidden; i++) {
            network.mutate(methods.mutation.ADD_NODE);
        }
        for (i = 0; i < connections - hidden; i++) {
            network.mutate(methods.mutation.ADD_CONN);
        }
        for (i = 0; i < backconnections; i++) {
            network.mutate(methods.mutation.ADD_BACK_CONN);
        }
        for (i = 0; i < selfconnections; i++) {
            network.mutate(methods.mutation.ADD_SELF_CONN);
        }
        for (i = 0; i < gates; i++) {
            network.mutate(methods.mutation.ADD_GATE);
        }
        return network;
    },
    LSTM: function (...args: any[]): Network {
        if (args.length < 3) {
            throw new Error('You have to specify at least 3 layers');
        }
        let last = args.pop();
        let outputLayer;
        if (typeof last === 'number') {
            outputLayer = new Group(last);
            last = {};
        } else {
            outputLayer = new Group(args.pop());
        }
        outputLayer.set({ type: 'output' });
        let options: any = {};
        options.memoryToMemory = last.memoryToMemory || false;
        options.outputToMemory = last.outputToMemory || false;
        options.outputToGates = last.outputToGates || false;
        options.inputToOutput = last.inputToOutput === undefined ? true : last.inputToOutput;
        options.inputToDeep = last.inputToDeep === undefined ? true : last.inputToDeep;
        let inputLayer = new Group(args.shift());
        inputLayer.set({ type: 'input' });
        let blocks = args;
        let nodes: any[] = [];
        nodes.push(inputLayer);
        let previous = inputLayer;
        for (let i = 0; i < blocks.length; i++) {
            let block = blocks[i];
            let inputGate = new Group(block);
            let forgetGate = new Group(block);
            let memoryCell = new Group(block);
            let outputGate = new Group(block);
            let outputBlock = i === blocks.length - 1 ? outputLayer : new Group(block);
            inputGate.set({ bias: 1 });
            forgetGate.set({ bias: 1 });
            outputGate.set({ bias: 1 });
            let input = previous.connect(memoryCell, methods.connection.ALL_TO_ALL);
            previous.connect(inputGate, methods.connection.ALL_TO_ALL);
            previous.connect(outputGate, methods.connection.ALL_TO_ALL);
            previous.connect(forgetGate, methods.connection.ALL_TO_ALL);
            memoryCell.connect(inputGate, methods.connection.ALL_TO_ALL);
            memoryCell.connect(forgetGate, methods.connection.ALL_TO_ALL);
            memoryCell.connect(outputGate, methods.connection.ALL_TO_ALL);
            let forget = memoryCell.connect(memoryCell, methods.connection.ONE_TO_ONE);
            let output = memoryCell.connect(outputBlock, methods.connection.ALL_TO_ALL);
            inputGate.gate(input, methods.gating.INPUT);
            forgetGate.gate(forget, methods.gating.SELF);
            outputGate.gate(output, methods.gating.OUTPUT);
            if (options.inputToDeep && i > 0) {
                let input = inputLayer.connect(memoryCell, methods.connection.ALL_TO_ALL);
                inputGate.gate(input, methods.gating.INPUT);
            }
            if (options.memoryToMemory) {
                let input = memoryCell.connect(memoryCell, methods.connection.ALL_TO_ELSE);
                inputGate.gate(input, methods.gating.INPUT);
            }
            if (options.outputToMemory) {
                let input = outputLayer.connect(memoryCell, methods.connection.ALL_TO_ALL);
                inputGate.gate(input, methods.gating.INPUT);
            }
            if (options.outputToGates) {
                outputLayer.connect(inputGate, methods.connection.ALL_TO_ALL);
                outputLayer.connect(forgetGate, methods.connection.ALL_TO_ALL);
                outputLayer.connect(outputGate, methods.connection.ALL_TO_ALL);
            }
            nodes.push(inputGate);
            nodes.push(forgetGate);
            nodes.push(memoryCell);
            nodes.push(outputGate);
            if (i !== blocks.length - 1) nodes.push(outputBlock);
            previous = outputBlock;
        }
        if (options.inputToOutput) {
            inputLayer.connect(outputLayer, methods.connection.ALL_TO_ALL);
        }
        nodes.push(outputLayer);
        return architect.Construct(nodes);
    },
    GRU: function (...args: any[]): Network {
        if (args.length < 3) {
            throw new Error('not enough layers (minimum 3) !!');
        }
        let inputLayer = new Group(args.shift());
        let outputLayer = new Group(args.pop());
        let blocks = args;
        let nodes: (Group | Layer)[] = [];
        nodes.push(inputLayer);
        let previous: Group | Layer = inputLayer;
        for (let i = 0; i < blocks.length; i++) {
            let layer = Layer.GRU(blocks[i]);
            previous.connect(layer);
            previous = layer;
            nodes.push(layer);
        }
        previous.connect(outputLayer);
        nodes.push(outputLayer);
        return architect.Construct(nodes);
    },
    Hopfield: function (size: number): Network {
        let input = new Group(size);
        let output = new Group(size);
        input.connect(output, methods.connection.ALL_TO_ALL);
        input.set({ type: 'input' });
        output.set({ squash: methods.activation.STEP, type: 'output' });
        let network = architect.Construct([input, output]);
        return network;
    },
    NARX: function (inputSize: number, hiddenLayers: number[] | number, outputSize: number, previousInput: number, previousOutput: number): Network {
        if (!Array.isArray(hiddenLayers)) {
            hiddenLayers = [hiddenLayers];
        }
        let nodes: any[] = [];
        let input = Layer.Dense(inputSize);
        let inputMemory = Layer.Memory(inputSize, previousInput);
        let hidden: any[] = [];
        let output = Layer.Dense(outputSize);
        let outputMemory = Layer.Memory(outputSize, previousOutput);
        nodes.push(input);
        nodes.push(outputMemory);
        for (let i = 0; i < hiddenLayers.length; i++) {
            let hiddenLayer = Layer.Dense(hiddenLayers[i]);
            hidden.push(hiddenLayer);
            nodes.push(hiddenLayer);
            if (typeof hidden[i - 1] !== 'undefined') {
                hidden[i - 1].connect(hiddenLayer, methods.connection.ALL_TO_ALL);
            }
        }
        nodes.push(inputMemory);
        nodes.push(output);
        input.connect(hidden[0], methods.connection.ALL_TO_ALL);
        input.connect(inputMemory, methods.connection.ONE_TO_ONE, 1);
        inputMemory.connect(hidden[0], methods.connection.ALL_TO_ALL);
        hidden[hidden.length - 1].connect(output, methods.connection.ALL_TO_ALL);
        output.connect(outputMemory, methods.connection.ONE_TO_ONE, 1);
        outputMemory.connect(hidden[0], methods.connection.ALL_TO_ALL);
        input.set({ type: 'input' });
        output.set({ type: 'output' });
        return architect.Construct(nodes);
    }
};

export default architect;
