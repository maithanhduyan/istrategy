// Converted from js/architecture/layer.js to TypeScript
import methods from '../methods/methods';
import Group from './group';
import Node from './node';

/**
 * Represents a layer in a neural network.
 */
class Layer {
    output: Group | null = null;
    nodes: (Group | Node)[] = [];
    connections: {
        in: any[];
        out: any[];
        self: any[];
    } = { in: [], out: [], self: [] };
    input?: (from: any, method?: any, weight?: number) => any;

    constructor() {
        this.output = null;
        this.nodes = [];
        this.connections = { in: [], out: [], self: [] };
    }

    activate(value?: number[]): number[] {
        let values: number[] = [];
        if (typeof value !== 'undefined' && value.length !== this.nodes.length) {
            throw new Error('Array with values should be same as the amount of nodes!');
        }
        for (let i = 0; i < this.nodes.length; i++) {
            let activation;
            if (typeof value === 'undefined') {
                activation = (this.nodes[i] as Node).activate();
            } else {
                activation = (this.nodes[i] as Node).activate(value[i]);
            }
            values.push(activation);
        }
        return values;
    }

    propagate(rate?: number, momentum?: number, target?: number[]): void {
        if (typeof target !== 'undefined' && target.length !== this.nodes.length) {
            throw new Error('Array with values should be same as the amount of nodes!');
        }
        for (let i = this.nodes.length - 1; i >= 0; i--) {
            if (typeof target === 'undefined') {
                (this.nodes[i] as Node).propagate(rate, momentum, true);
            } else {
                (this.nodes[i] as Node).propagate(rate, momentum, true, target[i]);
            }
        }
    }

    connect(target: Group | Node | Layer, method?: any, weight?: number): any {
        let connections;
        if (target instanceof Group || target instanceof Node) {
            connections = (this.output as Group).connect(target, method, weight);
        } else if (target instanceof Layer) {
            connections = target.input!(this, method, weight);
        }
        return connections;
    }

    gate(connections: any[] | any, method: any): void {
        (this.output as Group).gate(connections, method);
    }

    set(values: { bias?: number; squash?: any; type?: any }): void {
        for (let i = 0; i < this.nodes.length; i++) {
            let node = this.nodes[i];
            if (node instanceof Node) {
                if (typeof values.bias !== 'undefined') {
                    node.bias = values.bias;
                }
                node.squash = values.squash || node.squash;
                node.type = values.type || node.type;
            } else if (node instanceof Group) {
                node.set(values);
            }
        }
    }

    disconnect(target: Group | Node, twosided?: boolean): void {
        twosided = twosided || false;
        let i: number, j: number, k: number;
        if (target instanceof Group) {
            for (i = 0; i < this.nodes.length; i++) {
                for (j = 0; j < target.nodes.length; j++) {
                    (this.nodes[i] as Node).disconnect(target.nodes[j], twosided);
                    for (k = this.connections.out.length - 1; k >= 0; k--) {
                        let conn = this.connections.out[k];
                        if (conn.from === this.nodes[i] && conn.to === target.nodes[j]) {
                            this.connections.out.splice(k, 1);
                            break;
                        }
                    }
                    if (twosided) {
                        for (k = this.connections.in.length - 1; k >= 0; k--) {
                            let conn = this.connections.in[k];
                            if (conn.from === target.nodes[j] && conn.to === this.nodes[i]) {
                                this.connections.in.splice(k, 1);
                                break;
                            }
                        }
                    }
                }
            }
        } else if (target instanceof Node) {
            for (i = 0; i < this.nodes.length; i++) {
                (this.nodes[i] as Node).disconnect(target, twosided);
                for (j = this.connections.out.length - 1; j >= 0; j--) {
                    let conn = this.connections.out[j];
                    if (conn.from === this.nodes[i] && conn.to === target) {
                        this.connections.out.splice(j, 1);
                        break;
                    }
                }
                if (twosided) {
                    for (k = this.connections.in.length - 1; k >= 0; k--) {
                        let conn = this.connections.in[k];
                        if (conn.from === target && conn.to === this.nodes[i]) {
                            this.connections.in.splice(k, 1);
                            break;
                        }
                    }
                }
            }
        }
    }

    clear(): void {
        for (let i = 0; i < this.nodes.length; i++) {
            (this.nodes[i] as Node).clear();
        }
    }

    static Dense(size: number): Layer {
        let layer = new Layer();
        let block = new Group(size);
        layer.nodes.push(block);
        layer.output = block;
        layer.input = function (from: any, method?: any, weight?: number) {
            if (from instanceof Layer) from = from.output;
            method = method || methods.connection.ALL_TO_ALL;
            return from.connect(block, method, weight);
        };
        return layer;
    }

    static LSTM(size: number): Layer {
        let layer = new Layer();
        let inputGate = new Group(size);
        let forgetGate = new Group(size);
        let memoryCell = new Group(size);
        let outputGate = new Group(size);
        let outputBlock = new Group(size);
        inputGate.set({ bias: 1 });
        forgetGate.set({ bias: 1 });
        outputGate.set({ bias: 1 });
        memoryCell.connect(inputGate, methods.connection.ALL_TO_ALL);
        memoryCell.connect(forgetGate, methods.connection.ALL_TO_ALL);
        memoryCell.connect(outputGate, methods.connection.ALL_TO_ALL);
        let forget = memoryCell.connect(memoryCell, methods.connection.ONE_TO_ONE);
        let output = memoryCell.connect(outputBlock, methods.connection.ALL_TO_ALL);
        forgetGate.gate(forget, methods.gating.SELF);
        outputGate.gate(output, methods.gating.OUTPUT);
        layer.nodes = [inputGate, forgetGate, memoryCell, outputGate, outputBlock];
        layer.output = outputBlock;
        layer.input = function (from: any, method?: any, weight?: number) {
            if (from instanceof Layer) from = from.output;
            method = method || methods.connection.ALL_TO_ALL;
            let connections: any[] = [];
            let input = from.connect(memoryCell, method, weight);
            connections = connections.concat(input);
            connections = connections.concat(from.connect(inputGate, method, weight));
            connections = connections.concat(from.connect(outputGate, method, weight));
            connections = connections.concat(from.connect(forgetGate, method, weight));
            inputGate.gate(input, methods.gating.INPUT);
            return connections;
        };
        return layer;
    }

    static GRU(size: number): Layer {
        let layer = new Layer();
        let updateGate = new Group(size);
        let inverseUpdateGate = new Group(size);
        let resetGate = new Group(size);
        let memoryCell = new Group(size);
        let output = new Group(size);
        let previousOutput = new Group(size);
        previousOutput.set({ bias: 0, squash: methods.activation.IDENTITY, type: 'constant' });
        memoryCell.set({ squash: methods.activation.TANH });
        inverseUpdateGate.set({ bias: 0, squash: methods.activation.INVERSE, type: 'constant' });
        updateGate.set({ bias: 1 });
        resetGate.set({ bias: 0 });
        previousOutput.connect(updateGate, methods.connection.ALL_TO_ALL);
        updateGate.connect(inverseUpdateGate, methods.connection.ONE_TO_ONE, 1);
        previousOutput.connect(resetGate, methods.connection.ALL_TO_ALL);
        let reset = previousOutput.connect(memoryCell, methods.connection.ALL_TO_ALL);
        resetGate.gate(reset, methods.gating.OUTPUT);
        let update1 = previousOutput.connect(output, methods.connection.ALL_TO_ALL);
        let update2 = memoryCell.connect(output, methods.connection.ALL_TO_ALL);
        updateGate.gate(update1, methods.gating.OUTPUT);
        inverseUpdateGate.gate(update2, methods.gating.OUTPUT);
        output.connect(previousOutput, methods.connection.ONE_TO_ONE, 1);
        layer.nodes = [updateGate, inverseUpdateGate, resetGate, memoryCell, output, previousOutput];
        layer.output = output;
        layer.input = function (from: any, method?: any, weight?: number) {
            if (from instanceof Layer) from = from.output;
            method = method || methods.connection.ALL_TO_ALL;
            let connections: any[] = [];
            connections = connections.concat(from.connect(updateGate, method, weight));
            connections = connections.concat(from.connect(resetGate, method, weight));
            connections = connections.concat(from.connect(memoryCell, method, weight));
            return connections;
        };
        return layer;
    }

    static Memory(size: number, memory: number): Layer {
        let layer = new Layer();
        let previous: Group | null = null;
        let i: number;
        for (i = 0; i < memory; i++) {
            let block = new Group(size);
            block.set({ squash: methods.activation.IDENTITY, bias: 0, type: 'constant' });
            if (previous != null) {
                previous.connect(block, methods.connection.ONE_TO_ONE, 1);
            }
            layer.nodes.push(block);
            previous = block;
        }
        layer.nodes.reverse();
        for (i = 0; i < layer.nodes.length; i++) {
            (layer.nodes[i] as Group).nodes.reverse();
        }
        let outputGroup = new Group(0);
        for (let group in layer.nodes) {
            outputGroup.nodes = outputGroup.nodes.concat((layer.nodes[group] as Group).nodes);
        }
        layer.output = outputGroup;
        layer.input = function (from: any) {
            if (from instanceof Layer) from = from.output;
            if (from.nodes.length !== (layer.nodes[layer.nodes.length - 1] as Group).nodes.length) {
                throw new Error('Previous layer size must be same as memory size');
            }
            return from.connect(layer.nodes[layer.nodes.length - 1], methods.connection.ONE_TO_ONE, 1);
        };
        return layer;
    }
}

export default Layer;
